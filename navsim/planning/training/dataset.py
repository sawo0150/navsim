import gzip
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!" 
        # cache_path가 유효한 디렉토리인지 확인 - 아니면 assert 실패로 프로그램 멈춤
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]
        # log_names가 주어지면 → 그 중 실제로 존재하는 로그 디렉토리만 사용
        # 없으면 → 캐시 경로 안의 모든 로그 폴더를 자동으로 사용

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        # builder들은 .gz 파일명과 내부 포맷 정의에 쓰임

        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        # _load_valid_caches()를 호출해서 실제 .gz가 다 존재하는 token만 수집

        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """
        # 주어진 로그 이름 목록을 순회하며
        # 각 log 디렉토리 아래의 token 폴더를 탐색함
        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
            # log 폴더 내에서 각 token 폴더 확인 (예: token1234)
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                # 각 builder 이름에 해당하는 .gz 파일이 존재하는지 확인
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path
                # 모든 .gz가 다 있으면 → 유효한 캐시로 인정 → valid_cache_paths에 등록

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """
        # _cache_scene_with_token 역과정

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)
            # 각 builder에 대응하는 .gz 파일에서 feature tensor들을 불러와 딕셔너리에 추가

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)
            # target builder들도 동일한 방식으로 처리

        return (features, targets)

# SceneLoader	로그(.pkl)를 읽고, scene 단위로 잘라서 token별로 scene을 관리
# Dataset	SceneLoader가 제공한 token으로부터 feature/target을 생성하고, 학습 데이터로 제공
# cache_path	feature/target을 미리 저장해두는 디렉토리
# feature_builders, target_builders	각 scene에서 학습에 필요한 정보를 추출하는 모듈들
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches( # Data 불러오는법1 - cache 불러오기
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset() # Data 불러오는법2 - cache 생성

    @staticmethod
    # @staticmethod란는 클래스 내부에 있지만 클래스 인스턴스(self)에 접근하지 않는 메서드
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """
        # 이 함수는 주어진 캐시 경로 안에 있는 파일들을 확인해서,
        # 모든 feature와 target에 해당하는 .gz 파일이 있는지 확인하고,
        # 있으면 _valid_cache_paths[token] = 해당 경로로 추가함.

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():   # 캐시 경로가 있고 디렉토리일 경우에만 진행
            for log_path in cache_path.iterdir():   # 로그 단위로 순회
                for token_path in log_path.iterdir(): # 로그 폴더 안의 토큰별 폴더를 순회
                    found_caches: List[bool] = [] # "각 builder가 요구하는 .gz 파일이 존재하는지"를 저장할 리스트
                    for builder in feature_builders + target_builders: # feature_builders와 target_builders를 합쳐서 하나씩 반복
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())   # 파일이 있는지 확인
                    if all(found_caches):   # 모두 존재해야 유효한 캐시 - found_caches가 [True, True, True]라면 → all()은 True
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        # scene_loader에서 token에 해당하는 scene을 가져오고
        agent_input = scene.get_agent_input()   #return AgentInput(ego_statuses, cameras, lidars)
        #  AgentInput은 학습 feature를 생성할 때 필요한 최소 단위 (Ego의 pose, camera, lidar 데이터).
        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        # SceneMetadata에서 이 scene이 어느 로그(log_name) 소속인지, 토큰은 뭔지 알아냄.
        # 디렉토리 구조: cache_path / log_name / token

        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)   # Dict[str, Tensor]가 return됨
            dump_feature_target_to_pickle(data_dict_path, data_dict) # dict를 .gz 파일로 저장
            # feature를 만드는 builder들을 하나씩 순회하면서,
            # compute_features(agent_input) 호출하여 결과를 계산
            # 그 결과를 .gz 파일로 저장

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """
        # _cache_scene_with_token 역과정

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""
        # 캐시가 없을 경우 모든 scene을 직접 계산해서 .gz 파일로 저장하는 역할

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        # 무조건 새로 캐싱 (force_cache_computation=True) - 캐시가 이미 있더라도 무시하고 전부 다시 계산함.
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            # SceneLoader가 가진 전체 토큰 목록 중에서 아직 _valid_cache_paths에 등록되지 않은 토큰만 골라냄.
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
                # 캐싱 개수가 많을 때 경고 메시지를 알려줌. 학습 중에 캐싱하려 하지 말고, 따로 캐싱 스크립트를 사용하라는 권장 사항.
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)
            # 각각의 token에 대해 _cache_scene_with_token(token) 호출

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)
        # 전체 데이터셋의 길이를 반환함. 즉, 몇 개의 sample(token)이 있는지를 알려줘.
        # self._scene_loader.tokens 리스트의 길이와 같아.
        # 학습 중에 PyTorch가 배치 개수를 계산할 때 이걸 기반으로 삼음.

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        # 현재 인덱스(idx)에 해당하는 scene의 token을 가져오기
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"
            # cache_path=None으로 Dataset을 만들면:
            # 캐시 로딩은 안 되고,
            # __getitem__()에서 실시간 계산으로 처리함.
            # 하지만 cache_dataset()을 부르면 assert에 걸려서 💥멈춰버림.
            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)
