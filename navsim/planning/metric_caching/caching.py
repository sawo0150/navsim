import gc
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hydra.utils import instantiate
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

logger = logging.getLogger(__name__)


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """

    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[Path, DictConfig]]]) -> List[CacheResult]:
        def cache_single_scenario(
            scene_dict: Dict[str, Any], processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            scene = Scene.from_scene_dict_list(
                scene_dict,
                None,
                num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,
                num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,
                sensor_config=SensorConfig.build_no_sensors(),
            )
            # 👉 목적:
            # raw frame 데이터 (scene_dict)를 받아서 하나의 Scene 객체로 구성
            # ✅ 내부적으로 포함된 데이터:
            #  - ego 차량의 과거/미래 상태
            #  - 주변 객체의 상태들
            #  - GT 경로 정보
            #  - 센서 정보는 없음 (build_no_sensors())
            #  ==> 이 단계는 모델 추론이 아니라, 평가 기준 정보만 생성
            #  ==> 캐싱 단계에서는 ego GT trajectory와 지도, 경로, 객체 정보 등 평가 기준만 준비함
            #  ==> 모델이 입력으로 쓰는 센서 데이터(camera, lidar, radar 등)는 필요 없음

            scenario = NavSimScenario(
                scene,
                map_root=os.environ["NUPLAN_MAPS_ROOT"],
                map_version="nuplan-maps-v1.0",
            )
            # NavSimScenario는 Scene을 nuPlan 프레임워크의 시뮬레이션 / 평가 시스템이 이해할 수 있는 인터페이스로 바꾸기 위해 사용됨
            # nuPlan에서 통일된 평가/시뮬레이션 처리 방식에 맞추기 위해 만든 Adapter Layer

            # 🧪 예: 왜 Scene만으론 부족할까?
            # planner_input = PlannerInput(
            #     iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            #     history=...,
            #     traffic_light_data=scenario.get_traffic_light_status_at_iteration(0),
            # )
            # 이런 코드에서 scenario는 반드시 AbstractScenario를 따라야 합니다.
            # 즉, get_traffic_light_status_at_iteration() 같은 메서드가 반드시 구현돼야 하죠.

            # → Scene에는 그런 메서드가 없기 때문에, NavSimScenario로 감싸야만 플래너/평가기에서 사용 가능합니다.

            return processor.compute_and_save_metric_cache(scenario)

        def cache_single_synthetic_scenario(
            scene_path: Path, processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            # synthetic scenario : Original Scene과 다르게 로그로 생성하는게 아니라 로그 기반 합성 시뮬레이션 된 scenario임
            scene = Scene.load_from_disk(scene_path, None, SensorConfig.build_no_sensors())
            # synthetic scene은 .pkl.xz로 저장된 단일 파일을 디스크에서 직접 로딩함 (load_from_disk)
            # Scene.from_scene_dict_list(...)처럼 memory 상의 프레임 리스트가 아님

            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")

            return processor.compute_and_save_metric_cache(scenario)
            #         🧠 왜 따로 처리할까?
            # 1. 파일 구조가 다름
            # - original scene은 로그 전체를 로드하고 거기서 slicing해서 scene 생성
            # - synthetic scene은 이미 완성된 scene 객체 .pkl.xz 파일로 저장됨
            # → 구조적으로 로딩 방식이 완전히 다름

            # 2. 데이터 구성 차이
            # - synthetic scene은 기존에 없던 상황 (위험 상황, 보행자 충돌 등)을 합성한 것
            # - 따라서 실제 미래 trajectory (GT)가 존재하지 않을 수도 있음
            # - 이 때문에 human_trajectory = None으로 처리

            # 3. 구분해서 캐시 관리
            # - 실험적으로도 original/synthetic을 따로 분석하거나 비교해야 하므로
            # - SceneFrameType.ORIGINAL vs. SYNTHETIC로 구분하여 캐시에 태그를 붙임

        # 1️⃣ 내부 준비
        # args : data_points에서 꺼내온 것 - token과 log, hydra 설정 데이터 존재.
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        log_names = [a["log_file"] for a in args]
        tokens = [t for a in args for t in a["tokens"]]
        cfg: DictConfig = args[0]["cfg"]
        # 분산 시스템에서 어떤 노드/스레드에서 처리 중인지 식별
        # 처리할 로그 파일명과 토큰 목록 수집
        # cfg는 Hydra 설정 객체

        # 2️⃣ Scene 필터링 및 로딩
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        # SceneFilter를 설정값 기반으로 구성 - 맨처음 넣을때 넣음
        # TRAIN_TEST_SPLIT=navtest으로 sh파일에 넣음
        scene_filter.log_names = log_names  # <-- 워커마다 독립적인 로그 파일 목록 (앞에서 정의한 것)
        scene_filter.tokens = tokens  # <-- 워커마다 독립적인 토큰 목록 (앞에서 정의한 것)
        scene_loader = SceneLoader(
            synthetic_sensor_path=None,
            original_sensor_path=None,
            data_path=Path(cfg.navsim_log_path),
            synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        # 앞에서 이미 한번 정의했지만, 또 sceneloader 정의
        # 이유: 병렬 워커마다 독립적인 SceneLoader가 필요하기 때문
        #   - 각 워커는 자신에게 할당된 log_file + token만 필터링해서 처리해야 하기 때문임
        #   - 이때 사용하는 scene_filter.tokens는 해당 워커 전용 token만 포함하도록 설정되어 있음.

        # 3️⃣ MetricCacheProcessor 준비
        # Create feature preprocessor
        assert cfg.metric_cache_path is not None, f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"
        processor = MetricCacheProcessor(
            cache_path=cfg.metric_cache_path,
            force_feature_computation=cfg.force_feature_computation,
            proposal_sampling=instantiate(cfg.proposal_sampling),
        )
        # 캐시 결과를 저장할 디렉토리, 재계산 여부, trajectory 샘플링 방식 등 초기화
        # 실제 캐싱을 담당하는 주체

        logger.info(f"Extracted {len(scene_loader)} scenarios for thread_id={thread_id}, node_id={node_id}.")
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        
        # 4️⃣ Original Scene 캐싱 루프
        # 대상 : original scene
        # 설명 : .pkl 기반의 실제 자율주행 로그에서 나온 시나리오
        for idx, scene_dict in enumerate(scene_loader.scene_frames_dicts.values()): # 하나의 scene 불러옴
            logger.info(
                f"Processing scenario {idx + 1} / {len(scene_loader.scene_frames_dicts)} in thread_id={thread_id}, node_id={node_id}"
            )
            file_cache_metadata = cache_single_scenario(scene_dict, processor) # 하나의 scene 캐싱
            gc.collect()

            # 성공/실패 카운팅
            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        # 5️⃣ Synthetic Scene 캐싱 루프
        # 대상 : 합성(synthetic) scene
        # 설명 : 위험 상황 등을 인위적으로 만든 시나리오
        for idx, (scene_path, _) in enumerate(scene_loader.synthetic_scenes.values()):
            logger.info(
                f"Processing synthetic scenario {idx + 1} / {len(scene_loader.synthetic_scenes)} in thread_id={thread_id}, node_id={node_id}"
            )
            file_cache_metadata = cache_single_synthetic_scenario(scene_path, processor)
            gc.collect()

            # 성공/실패 카운팅
            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        # 6️⃣ 캐시 결과 수집 및 반환
        logger.info(f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}")
        return [
            CacheResult(
                failures=num_failures,
                successes=num_successes,
                cache_metadata=all_file_cache_metadata,
            )
        ]
    # 성공/실패 개수, 캐시된 메타데이터 목록을 담은 CacheResult 리스트 반환

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()
# 내부 함수가 끝난 뒤 스코프 전체가 종료되므로,
# 내부에서 생성된 큰 객체들(SceneLoader, Scene, NavSimScenario 등)이 완전히 해제될 수 있음

# 그리고 마지막에 gc.collect()를 호출해서 수동으로 가비지 컬렉션을 강제 실행합니다.


    # 실제로는 cache_scenarios() 안에 또 cache_scenarios_internal()을 
    # 중첩 함수(nested function)로 정의함

# cache_scenarios() 내부에 cache_scenarios_internal()을 둔 건
# **“병렬 환경에서 대용량 데이터를 메모리 누수 없이 안전하게 처리하기 위한 설계”**입니다.

# 이건 Python에서 대규모 데이터 처리 시 꽤 흔한 고급 테크닉이에요.

    return result


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    # SceneLoader로 데이터를 불러옴 → 시나리오별로 쪼갬
    # 각 시나리오를 worker를 통해 병렬 처리 → cache_scenarios()에 넘김
    # 각 시나리오에서 MetricCacheProcessor를 이용해 메트릭을 계산하고 저장
    # 캐싱 성공/실패 정보를 종합해 로그로 출력 + 메타데이터 CSV로 저장

    assert cfg.metric_cache_path is not None, f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here

    # 1️⃣ SceneLoader 초기화
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    # SceneLoader는 .pkl 로그 데이터를 로드해서 scene 단위로 나눔.
    # scene_filter에 따라 조건(예: 프레임 수, 경로 포함 여부 등)을 만족하는 scene만 필터링.
    # 원본(scene_frames_dicts)과 합성(synthetic_scenes) 둘 다 처리 가능.

    # 2️⃣ 작업 분할 (token 기준)
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    # 각 log 파일별로 포함된 scene token 리스트를 추출하고, 
    # 이 token 단위로 병렬 처리할 수 있도록 data_points 리스트를 만듦.

    # 3️⃣ 병렬 캐싱 실행
    logger.info("Starting metric caching of %s files...", str(len(data_points)))

    # 4️⃣ 내부적으로 캐싱되는 내용 (핵심) - cache_scenarios함수
    cache_results = worker_map(worker, cache_scenarios, data_points)
    # Ray나 multiprocessing 기반 워커를 통해 cache_scenarios()를 병렬로 실행.
    # 각 작업은 Scene을 로딩하고 → NavSimScenario로 감싸고 → MetricCacheProcessor로 캐싱.

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    if num_fail == 0:
        logger.info(
            "Completed dataset caching! All %s features and targets were cached successfully.",
            str(num_total),
        )
    else:
        logger.info(
            "Completed dataset caching! Failed features and targets: %s out of %s",
            str(num_fail),
            str(num_total),
        )

    # 5️⃣ 메타데이터 기록
    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]
    # 캐시가 성공한 시나리오들의 메타데이터(csv)가 metric_cache_path/metadata/에 저장됨.
    # 이 csv 파일은 이후 학습 또는 평가 때 어떤 데이터가 유효한지 필터링할 때 사용.

    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
    save_cache_metadata(cached_metadata, Path(cfg.metric_cache_path), node_id)
    logger.info("Done storing metadata csv file.")

    # 최종 캐시 파일 구조
    # metric_cache/
    # ├── metadata/
    # │   └── cache_node_0.csv  ← 어떤 scene들이 성공적으로 캐싱됐는지 기록
    # ├── token_xxxxxxxxxx.pkl.xz  ← MetricCache 객체 (scene 단위 캐시)
    # ├── token_yyyyyyyyyy.pkl.xz
...
