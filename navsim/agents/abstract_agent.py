from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

# AbstractAgent는 추상 클래스 (abstract class) 라고 부르고, 설계의 뼈대 같은 존재
# agent의 공통적인 interface를 정의하는 추상 클래스
# 모든 agent는 이 클래스를 상속받아야 함
class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling, # type hint : 변수 이름 - 타입
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene
        self._trajectory_sampling = trajectory_sampling
        #  TrajectorySampling은 "얼마나 긴 궤적을", "얼마나 촘촘하게" 샘플링할지 정의하는 클래스
        # 왜 필요한가? - TrajectoryTargetBuilder에서 이걸 사용해서 target trajectory의 길이를 정하고
        # MLP 출력의 차원 수도 이걸로 맞춰짐!
        # → Linear(..., self._trajectory_sampling.num_poses * 3) ← 여기서!

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """
        # 어떤 sensor를 사용하는지 정의

    @abstractmethod # 반드시 하위 클래스에서 구현해야하는 함수 표시 데코레이터.
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """
        # 초기화 작업 (ex. checkpoint 불러오기)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        # 모델 추론 작업 정의
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        # 학습에 필요한 feature builder 정의
        #feature : 모델이 학습할때 입력으로 주는 데이터 (ex. 차량속도, 가속도, 조향 명령어)
        # Dataloader에서 호출할때 이 builder를 호출.

        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        # 학습에 필요한 target builder 정의
        # target : 모델이 학습할때 맞춰야 할 정답 데이터 (ex. 미래 주행 궤적 - x, y, heading)

        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        # AgentInput, Trajectory : repo 내에서 정의하는 class
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        # agent가 실제로 예측할때 사용하는 메서드
        self.eval()
        # pytorch 모델 평가 모드로 설정
        # - Dropout(과적합 방지 위해 일부 뉴런 꺼버리는) 비활성화, BatchNorm 고정(훈련중 저장해둔 mean, variance)
        # - 추론할 때는 **torch.no_grad()**도 같이 쓰는 게 국룰

        features: Dict[str, torch.Tensor] = {} # type hint : 변수 이름 - 타입(dict, key: str, value: torch.Tensor)
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # FeatureBuilder를 통해 입력(agent_input)으로부터 feature를 만듦
        # (예: velocity + acceleration + command → "ego_status")
        # dict1.update(dict2) : dict2의 내용을 dict1에 병합(merge) 하는 함수
        # dict1이 수정되고, dict2의 키-값이 추가되거나 덮어써짐 - 같은 키가 있으면, 뒤에 있는 게 우선


        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        # 학습/추론은 보통 batch로 하니까, [batch, feature] 형태로 만듦
        # 여기선 batch size = 1

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses, self._trajectory_sampling)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        # 이 메서드는 학습할 때 사용할 Optimizer를 반환하는 함수
        # PyTorch Lightning은 학습 시작할 때, 이걸 호출해서 Optimizer를 세팅

        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        # 콜백이란?
        # 에포크가 끝날 때 뭔가를 한다거나
        # 특정 조건에서 early stopping 한다거나
        # 모델 체크포인트 저장한다거나
        # 이런 걸 자동으로 처리해주는 후크(hook) 
        return []
