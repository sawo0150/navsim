from __future__ import annotations

import lzma
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache

FrameList = List[Dict[str, Any]]

# <데이터 관계>
# 🔄 관계로 설명하자면…
# 하나의 Log 안에는 수천 개의 Frame이 있음
# 이 Frame들을 일정 간격으로 잘라서 여러 개의 Scene을 만들 수 있음
# 각 Scene은 특정 위치의 프레임 기준으로 Token을 갖게 됨
# 이 Token은 학습 시 "어떤 scene이었는지" 고유하게 추적하는 ID
# 예. 
# 🗂️ log_001.pkl
#  └── frame_0
#  └── frame_1
#  ...
#  └── frame_1999
# → scene_01 = frame_0 ~ frame_14 → token = frame_4.token
# → scene_02 = frame_10 ~ frame_24 → token = frame_14.token

#  token은 "scene의 ID"이자, log 속 위치 정보를 내포한 이름표
# 그런데 이 token은 결국 log 안의 프레임 ID이기도 함

# → 결국 token이 다르면 scene 자체도 다르고,
# token만 보면 어느 log에서 나온 scene인지도 추적 가능

def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Tuple[Dict[str, FrameList], List[str]]:
    """
    Load a set of scenes from dataset, while applying scene filter configuration.
    :param data_path: root directory of log folder
    :param scene_filter: scene filtering configuration class
    :return: dictionary of raw logs format, and list of final frame tokens that can be used to filter synthetic scenes
    """
    # 데이터 폴더 내의 로그 파일들을 읽어서, SceneFilter 조건에 맞는 scene들을 필터링해주는 함수

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        """Helper function to split frame list according to sampling specification."""
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]
    # 로그 하나는 수많은 프레임으로 구성되어 있는데, 이걸 일정한 길이(num_frames)로 잘라서 학습용 scene 단위로 나눠줌
    # frame_interval은 프레임 건너뛰기 간격

    filtered_scenes: Dict[str, Scene] = {}
    # keep track of the final frame tokens which refer to the original scene of potential second stage synthetic scenes
    final_frame_tokens: List[str] = []
    stop_loading: bool = False

    # filter logs
    log_files = list(data_path.iterdir())
    # data_path 폴더 내의 .pkl 로그 파일들을 전부 가져옴
    if scene_filter.log_names is not None:
        log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in scene_filter.log_names]
    # 만약 log_names 필터 조건이 있으면, 그 이름에 해당하는 로그만 선택함

    if scene_filter.tokens is not None:
        # 특정 scene token 기준으로 필터링할지 여부를 설정함
        filter_tokens = True
        tokens = set(scene_filter.tokens)
    else:
        filter_tokens = False

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        # 로그 하나하나 열어서 프레임 목록(scene_dict_list)을 불러오기

        for frame_list in split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
            # 각 scene 후보(frame_list)에 대해 아래 조건을 적용:

            # Filter scenes which are too short - 프레임 개수 부족하면 건너뜀
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route - route 정보 없는 scene 제외
            if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                continue

            # Filter by token-  token 필터링 적용
            token = frame_list[scene_filter.num_history_frames - 1]["token"] # ego frame을 scene token으로 사용함
            if filter_tokens and token not in tokens:
                continue

            # 필터 통과한 scene 저장
            filtered_scenes[token] = frame_list
            final_frame_token = frame_list[scene_filter.num_frames - 1]["token"]
            #  TODO: if num_future_frames > proposal_sampling frames, then the final_frame_token index is wrong
            final_frame_tokens.append(final_frame_token)

            # scene 최대 개수 제한 확인
            if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes, final_frame_tokens
# filter_scenes()는 .pkl 로그들을 읽고, SceneFilter 조건에 맞는 scene만 깔끔하게 추려서 리스트로 반환하는 scene 전처리기

# 🔶 synthetic_scenes란?
# SceneLoader 클래스에서 synthetic_scenes는 합성된(synthetic) 시뮬레이션 장면들을 담고 있는 딕셔너리.
# 이는 실제 자율주행 로그(log)로부터 만들어진 원본 데이터와는 달리, 기존의 원본 데이터를 바탕으로 인위적으로 생성된 데이터셋이라고 보면 됨.
# 이런 synthetic scene은 예를 들면 다음과 같은 경우에 쓰여:
# - 원본 주행에서는 발생하지 않은 위험 상황을 시뮬레이션으로 추가
# - 다양한 주행 조건(보행자 등장, 긴급 정지 등)을 실험하기 위해

def filter_synthetic_scenes(
    data_path: Path, scene_filter: SceneFilter, stage1_scenes_final_frames_tokens: List[str]
) -> Dict[str, Tuple[Path, str]]:
    # Load all the synthetic scenes that belong to the original scenes already loaded
    loaded_scenes: Dict[str, Tuple[Path, str, int]] = {}
    synthetic_scenes_paths = list(data_path.iterdir())
    # 합성 장면 폴더 내의 모든 파일들을 가져옴
    # 이걸 로드해야지만, 이런 정보들을 꺼내기 위해:
    # scene.scene_metadata.initial_token
    # scene.scene_metadata.log_name
    # scene.scene_metadata.corresponding_original_scene

    filter_logs = scene_filter.log_names is not None
    filter_tokens = scene_filter.synthetic_scene_tokens is not None

    for scene_path in tqdm(synthetic_scenes_paths, desc="Loading synthetic scenes"):
        synthetic_scene = Scene.load_from_disk(scene_path, None, None)

        # if a token is requested specifically, we load it even if it is not related to the original scenes loaded
        if filter_tokens and synthetic_scene.scene_metadata.initial_token not in scene_filter.synthetic_scene_tokens:
            continue

        # filter by log names
        log_name = synthetic_scene.scene_metadata.log_name
        if filter_logs and log_name not in scene_filter.log_names:
            continue

        # if we don't filter for tokens explicitly, we load only the synthetic scenes required to run a second stage for the original scenes loaded
        if (
            not filter_tokens
            and synthetic_scene.scene_metadata.corresponding_original_scene not in stage1_scenes_final_frames_tokens
        ):
            continue

        loaded_scenes.update({synthetic_scene.scene_metadata.initial_token: [scene_path, log_name]})

    return loaded_scenes

# @property는 클래스 메서드를 속성처럼 사용할 수 있게 만들어주는 데코레이터
class SceneLoader:
    """Simple data loader of scenes from logs."""
    # Scene 하나는 여러 프레임(시간)을 담고 있는 단위 데이터

    def __init__(
        self,
        data_path: Path, # log 폴더 경로
        original_sensor_path: Path, # 센서(camera, lidar, radar) 폴더 경로
        scene_filter: SceneFilter, # SceneFilter 객체
        synthetic_sensor_path: Path = None, # 합성 장면 센서 폴더 경로
        synthetic_scenes_path: Path = None, # 합성 장면 폴더 경로
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(), # 센서 설정
    ):
        """
        Initializes the scene data loader.
        :param data_path: root directory of log folder
        :param synthetic_sensor_path: root directory of sensor  (synthetic)
        :param original_sensor_path: root directory of sensor  (original)
        :param scene_filter: dataclass for scene filtering specification
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        """

        self.scene_frames_dicts, stage1_scenes_final_frames_tokens = filter_scenes(data_path, scene_filter)
        self._synthetic_sensor_path = synthetic_sensor_path
        self._original_sensor_path = original_sensor_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config

        if scene_filter.include_synthetic_scenes:
            assert (
                synthetic_scenes_path is not None
            ), "Synthetic scenes path cannot be None, when synthetic scenes_filter.include_synthetic_scenes is set to True."
            self.synthetic_scenes = filter_synthetic_scenes(
                data_path=synthetic_scenes_path,
                scene_filter=scene_filter,
                stage1_scenes_final_frames_tokens=stage1_scenes_final_frames_tokens,
            )
            self.synthetic_scenes_tokens = set(self.synthetic_scenes.keys())
        else:
            self.synthetic_scenes = {}
            self.synthetic_scenes_tokens = set()

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys()) + list(self.synthetic_scenes.keys())
        # 로드된 모든 scene들의 token 리스트를 반환함. original + synthetic 둘 다 포함됨.

    @property
    def tokens_stage_one(self) -> List[str]:
        """
        original scenes
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys())
        # 오직 original scenes만 가져옴.

    @property
    def reactive_tokens_stage_two(self) -> List[str]:
        """
        reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        reactive_synthetic_initial_tokens = self._scene_filter.reactive_synthetic_initial_tokens
        if reactive_synthetic_initial_tokens is None:
            return None
        return list(set(self.synthetic_scenes_tokens) & set(reactive_synthetic_initial_tokens))
        #   SceneFilter.reactive_synthetic_initial_tokens에 있는 것만 가져옴.
        # 즉, "2단계용 리액티브 synthetic scene"만 골라주는 역할.

    @property
    def non_reactive_tokens_stage_two(self) -> List[str]:
        """
        non reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        non_reactive_synthetic_initial_tokens = self._scene_filter.non_reactive_synthetic_initial_tokens
        if non_reactive_synthetic_initial_tokens is None:
            return None
        return list(set(self.synthetic_scenes_tokens) & set(non_reactive_synthetic_initial_tokens))
        # 위와 같은 방식으로 비리액티브 synthetic scene만 필터링.

    # 이렇게 나누는 이유는, stage-1 / stage-2 실험 분리 또는 학습 시 loss 조절 등에 유용하게 쓰이기 때문?

    @property
    def reactive_tokens(self) -> List[str]:
        """
        original scenes and reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        reactive_synthetic_initial_tokens = self._scene_filter.reactive_synthetic_initial_tokens
        if reactive_synthetic_initial_tokens is None:
            return list(self.scene_frames_dicts.keys())
        return list(self.scene_frames_dicts.keys()) + list(
            set(self.synthetic_scenes_tokens) & set(reactive_synthetic_initial_tokens)
        )

    @property
    def non_reactive_tokens(self) -> List[str]:
        """
        original scenes and non reactive synthetic scenes
        :return: list of scene identifiers for loading.
        """
        non_reactive_synthetic_initial_tokens = self._scene_filter.non_reactive_synthetic_initial_tokens
        if non_reactive_synthetic_initial_tokens is None:
            return list(self.scene_frames_dicts.keys())
        return list(self.scene_frames_dicts.keys()) + list(
            set(self.synthetic_scenes_tokens) & set(non_reactive_synthetic_initial_tokens)
        )

    def __len__(self) -> int:
        """
        :return: number for scenes possible to load.
        """
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        """
        :param idx: index of scene
        :return: unique scene identifier
        """
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        """
        Loads scene given a scene identifier string (token).
        :param token: scene identifier string.
        :return: scene dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],
                sensor_blobs_path=self._synthetic_sensor_path,
                sensor_config=self._sensor_config,
            )
        else:
            return Scene.from_scene_dict_list(
                self.scene_frames_dicts[token],
                self._original_sensor_path,
                num_history_frames=self._scene_filter.num_history_frames,
                num_future_frames=self._scene_filter.num_future_frames,
                sensor_config=self._sensor_config,
            )
        # synthetic이면 디스크에서 바로 읽어오고, original이면 이미 메모리에 있던 frame_list로 Scene 구성함.

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],
                sensor_blobs_path=self._synthetic_sensor_path,
                sensor_config=self._sensor_config,
            ).get_agent_input()
        else:
            return AgentInput.from_scene_dict_list(
                self.scene_frames_dicts[token],
                self._original_sensor_path,
                num_history_frames=self._scene_filter.num_history_frames,
                sensor_config=self._sensor_config,
            )
        # get_scene_from_token과 비슷하지만,
        # Scene 전체가 아니라, 그 안에서 AgentInput만 뽑아내서 리턴함.

        # 즉, 학습용 input만 추출할 때 쓰는 메서드.

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        Collect tokens for each logs file given filtering.
        :return: dictionary of logs names and tokens
        """
        # generate a dict that contains a list of tokens for each log-name
        tokens_per_logs: Dict[str, List[str]] = {}
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs.update({log_name: [token]})

        for scene_path, log_name in self.synthetic_scenes.values():
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(scene_path.stem)
            else:
                tokens_per_logs.update({log_name: [scene_path.stem]})

        return tokens_per_logs
    # 로그별로 어떤 token들이 포함됐는지 딕셔너리로 정리해줌.
    # 학습 로그 분포를 분석하거나, 특정 로그만 다시 검토할 때 유용함.
    # tokens_per_logs = {
    #   "log_001": ["token_a", "token_b", ...],
    #   "log_002": ["token_x", "token_y", ...],
    # }


class MetricCacheLoader:
    """Simple dataloader for metric cache."""

    def __init__(self, cache_path: Path, file_name: str = "metric_cache.pkl"):
        """
        Initializes the metric cache loader.
        :param cache_path: directory of cache folder
        :param file_name: file name of cached files, defaults to "metric_cache.pkl"
        """

        self._file_name = file_name
        self.metric_cache_paths = self._load_metric_cache_paths(cache_path)

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:
        """
        Helper function to load all cache file paths from folder.
        :param cache_path: directory of cache folder
        :return: dictionary of token and file path
        """
        metadata_dir = cache_path / "metadata"
        metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]
        with open(str(metadata_file), "r") as f:
            cache_paths = f.read().splitlines()[1:]
        metric_cache_dict = {cache_path.split("/")[-2]: cache_path for cache_path in cache_paths}
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        """
        :return: number for scenes possible to load.
        """
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        """
        :param idx: index of cache to cache to load
        :return: metric cache dataclass
        """
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:
        """
        Load metric cache from scene identifier
        :param token: unique identifier of scene
        :return: metric cache dataclass
        """
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        return metric_cache

    def to_pickle(self, path: Path) -> None:
        """
        Dumps complete metric cache into pickle.
        :param path: directory of cache folder
        """
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
