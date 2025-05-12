import logging

import hydra
from nuplan.planning.script.builders.logging_builder import build_logger
# 설정 파일(cfg)에 정의된 로깅 구성을 기반으로 로깅 시스템을 초기화함.

from omegaconf import DictConfig

from navsim.planning.metric_caching.caching import cache_data
# 설정 파일과 워커를 사용하여 시뮬레이션 데이터를 처리하고, 결과를 캐시에 저장함.
# 예시: 특정 시나리오에 대한 시뮬레이션 결과를 미리 계산하여 저장함으로써, 이후 분석이나 평가 시 빠르게 접근할 수 있습니다.

from navsim.planning.script.builders.worker_pool_builder import build_worker
# 설정 파일에 따라 워커를 구성함. 예를 들어, 멀티프로세싱 또는 분산 처리를 위한 워커를 생성할 수 있음.
# cfg.worker가 "sequential"이면 순차 처리 워커를, "ray_distributed"이면 Ray를 활용한 분산 처리 워커를 생성함.

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/metric_caching"
CONFIG_NAME = "default_metric_caching"
# hydra yaml 파일 : config/metric_caching/default_metric_caching.yaml
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for metric caching.
    :param cfg: omegaconf dictionary
    """
    # Configure logger
    build_logger(cfg)

    # Build worker
    worker = build_worker(cfg)

    # Precompute and cache all features
    logger.info("Starting Metric Caching...")
    if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
        raise AssertionError("ray in distributed mode will not work with this job")
    cache_data(cfg=cfg, worker=worker)


if __name__ == "__main__":
    main()
