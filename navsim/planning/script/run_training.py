import logging
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl #PyTorch Lightning은 PyTorch 반복되는 학습 코드를 자동화해주는 프레임워크
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    # 목적 : Hydra config와 Agent를 받아서, 학습/검증용 Dataset 객체 두 개를 만들어 반환하는 것
    # 1. SceneFilter 만들기
    # 2. SceneLoader 만들기
    # 3. Dataset 만들기 (여기서 features/targets 구성될 준비)
    # 4. 최종적으로 train_data, val_data 반환


    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    # scene_filter는 어떤 scene을 학습/검증에 쓸지 기준을 정해주는 역할
    # 예:
    # 몇 초 과거 / 미래를 사용할 건지
    # 어떤 로그 이름만 사용할 건지
    # synthetic scene 포함할 건지 등등…

    # default_training.yaml - default_common.yaml - train_test_split 설정 존재함
    # 이 설정은 train_test_split: ??? 로 되어 있어서 처음 실행할때 반드시 명시해야함
    # transfuser, mlp : navtrain으로 사용함

    # default_training.yaml - default_common.yaml - train_test_split 설정 존재함
    # 이 설정은 train_test_split: ??? 로 되어 있어서 처음 실행할때 반드시 명시해야함
    # transfuser, mlp : navtrain으로 사용함

    # navtrain.yaml 파일 존재함 (config/common/train_test_split/navtrain.yaml)
    # defaults:
    # - scene_filter: navtrain
    # data_split: trainval

    # scene_filter: navtrain는 또 common/train_test_split/scene_filter/navtrain.yaml 파일 존재함


    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data

# hydra 모듈 활용함 - 복잡한 설정 (학습률, 배치 사이즈, 최대 에폭 등) 쉽게 할 수 있음
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)    
# hydra config 파일 받아오기 : config/training/default_training.yaml --> 
def main(cfg: DictConfig) -> None: # type hint 문법 : cfg 타입은 DictConfig, 리턴값은 None
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True) # 모든 랜덤 시드 고정
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent) # type hint : agent라는 변수의 타입은 AbstractAgent(class) or 하위 class!!
    # type 강제는 아님
    # instantiate : hydra config 파일 중 agent 설정 부분 활용 class 객체 생성
    #     defaults:
    # - default_common
    # - default_evaluation
    # - default_train_val_test_log_split
    # - agent: ego_status_mlp_agent
    # - _self_
    # 여기서 config/agent/ego_status_mlp_agent.yaml 파일 활용해서 그걸 객체로 만들어줌

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )
    # PyTorch Lightning이 학습을 자동으로 관리하게 도와주는 "브릿지 클래스"
    # 🧠 Agent ← (wrap) → LightningModule ← (학습 관리) → Trainer
    # - PyTorch Lightning에서 학습 루프를 자동화하려면 반드시 `LightningModule`을 상속해야함
    # - 그러면 `training_step()`, `validation_step()` 등을 오버라이드할 수 있음

    if cfg.use_cache_without_dataset:
        # 캐시를 불러오기만 하는 경우 (CacheOnlyDataset 사용)
        # 이미 생성된 feature/target들을 .pt 파일 등으로 저장해뒀고,
        # 그걸 메모리에 다시 로딩만 해서 쓰는 구조
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        # cache를 학습할때 안쓰는 경우 - 기본?
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    # pytorch lightning trainer 객체 생성 : 학습 과정 관리
    # yaml 파일 중 trainer 설정 부분 활용

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
