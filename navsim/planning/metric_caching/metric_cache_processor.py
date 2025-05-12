import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry

from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MapParameters, MetricCache
from navsim.planning.metric_caching.metric_caching_utils import StateInterpolator
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy


class MetricCacheProcessor:
    """Class for creating metric cache in NAVSIM."""

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
        proposal_sampling: TrajectorySampling,
    ):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation

        # 1s additional observation for ttc metric
        future_poses = proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length)
        future_sampling = TrajectorySampling(num_poses=future_poses, interval_length=proposal_sampling.interval_length)
        self._proposal_sampling = proposal_sampling
        self._map_radius = 100

        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=future_sampling,
            proposal_sampling=self._proposal_sampling,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
                fallback_target_velocity=15.0,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.5,
                decel_max=3.0,
            ),
            lateral_offsets=[-1.0, 1.0],
            map_radius=self._map_radius,
        )

    def _get_planner_inputs(self, scenario: AbstractScenario) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,
            ego_states=[scenario.initial_ego_state],
            observations=[scenario.initial_tracked_objects],
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
        )

        return planner_input, planner_initialization

    def _interpolate_gt_observation(self, scenario: NavSimScenario) -> List[DetectionsTracks]:
        """
        Helper function to interpolate detections tracks to higher temporal resolution.
        :param scenario: scenario interface of nuPlan framework
        :return: interpolated detection tracks
        """

        # TODO: add to config
        state_size = 6  # (time, x, y, heading, velo_x, velo_y)

        time_horizon = self._proposal_sampling.time_horizon  # [s]
        resolution_step = 0.5  # [s]
        interpolate_step = self._proposal_sampling.interval_length  # [s]

        scenario_step = scenario.database_interval  # [s]

        # sample detection tracks a 2Hz
        relative_time_s = np.arange(0, (time_horizon * 1 / resolution_step) + 1, 1, dtype=float) * resolution_step

        gt_indices = np.arange(
            0,
            int(time_horizon / scenario_step) + 1,
            int(resolution_step / scenario_step),
        )
        gt_detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration) for iteration in gt_indices
        ]

        detection_tracks_states: Dict[str, Any] = {}
        unique_detection_tracks: Dict[str, Any] = {}

        for time_s, detection_track in zip(relative_time_s, gt_detection_tracks):

            for tracked_object in detection_track.tracked_objects:
                # log detection track
                token = tracked_object.track_token

                # extract states for dynamic and static objects
                tracked_state = np.zeros(state_size, dtype=np.float64)
                tracked_state[:4] = (
                    time_s,
                    tracked_object.center.x,
                    tracked_object.center.y,
                    tracked_object.center.heading,
                )

                if tracked_object.tracked_object_type in AGENT_TYPES:
                    # extract additional states for dynamic objects
                    tracked_state[4:] = (
                        tracked_object.velocity.x,
                        tracked_object.velocity.y,
                    )

                # found new object
                if token not in detection_tracks_states.keys():
                    detection_tracks_states[token] = [tracked_state]
                    unique_detection_tracks[token] = tracked_object

                # object already existed
                else:
                    detection_tracks_states[token].append(tracked_state)

        # create time interpolators
        detection_interpolators: Dict[str, StateInterpolator] = {}
        for token, states_list in detection_tracks_states.items():
            states = np.array(states_list, dtype=np.float64)
            detection_interpolators[token] = StateInterpolator(states)

        # interpolate at 10Hz
        interpolated_time_s = np.arange(0, int(time_horizon / interpolate_step) + 1, 1, dtype=float) * interpolate_step

        interpolated_detection_tracks = []
        for time_s in interpolated_time_s:
            interpolated_tracks = []
            for token, interpolator in detection_interpolators.items():
                initial_detection_track = unique_detection_tracks[token]
                interpolated_state = interpolator.interpolate(time_s)

                if interpolator.start_time == interpolator.end_time:
                    interpolated_tracks.append(initial_detection_track)

                elif interpolated_state is not None:

                    tracked_type = initial_detection_track.tracked_object_type
                    metadata = initial_detection_track.metadata  # copied since time stamp is ignored

                    oriented_box = OrientedBox(
                        StateSE2(*interpolated_state[:3]),
                        initial_detection_track.box.length,
                        initial_detection_track.box.width,
                        initial_detection_track.box.height,
                    )

                    if tracked_type in AGENT_TYPES:
                        velocity = StateVector2D(*interpolated_state[3:])

                        detection_track = Agent(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            velocity=velocity,
                            metadata=initial_detection_track.metadata,  # simply copy
                        )
                    else:
                        detection_track = StaticObject(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            metadata=metadata,
                        )

                    interpolated_tracks.append(detection_track)
            interpolated_detection_tracks.append(DetectionsTracks(TrackedObjects(interpolated_tracks)))
        return interpolated_detection_tracks

    def _build_pdm_observation(
        self,
        interpolated_detection_tracks: List[DetectionsTracks],
        interpolated_traffic_light_data: List[List[TrafficLightStatusData]],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ):
        # convert to pdm observation
        pdm_observation = PDMObservation(
            self._proposal_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
            extend_observation_for_ttc=False,
        )
        pdm_observation.update_detections_tracks(
            interpolated_detection_tracks,
            interpolated_traffic_light_data,
            route_lane_dict,
            compute_traffic_light_data=True,
        )
        return pdm_observation

    def _interpolate_traffic_light_status(self, scenario: NavSimScenario) -> List[List[TrafficLightStatusData]]:

        time_horizon = self._proposal_sampling.time_horizon  # [s]
        interpolate_step = self._proposal_sampling.interval_length  # [s]

        scenario_step = scenario.database_interval  # [s]
        gt_indices = np.arange(0, int(time_horizon / scenario_step) + 1, 1, dtype=int)

        traffic_light_status = []
        for iteration in gt_indices:
            current_status_list = list(scenario.get_traffic_light_status_at_iteration(iteration=iteration))
            for _ in range(int(scenario_step / interpolate_step)):
                traffic_light_status.append(current_status_list)

        if scenario_step == interpolate_step:
            return traffic_light_status
        else:
            return traffic_light_status[: -int(scenario_step / interpolate_step) + 1]

    def _load_route_dicts(
        self, scenario: NavSimScenario, route_roadblock_ids: List[str]
    ) -> Tuple[Dict[str, RoadBlockGraphEdgeMapObject], Dict[str, LaneGraphEdgeMapObject]]:
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        route_roadblock_dict = {}
        route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)

            route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                route_lane_dict[lane.id] = lane

        return route_roadblock_dict, route_lane_dict

    def _build_file_path(self, scenario: NavSimScenario) -> pathlib.Path:
        return (
            (self._cache_path / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl")
            if self._cache_path
            else None
        )

    def compute_and_save_metric_cache(self, scenario: NavSimScenario) -> Optional[CacheMetadataEntry]:
        # 이 함수 한 줄로 인해 "이 시나리오에 대해 평가에 필요한 모든 기준 정보"가 캐시 파일로 생성됨.
        # 이후에는 모델이 만든 예측 경로만 넣어서 메트릭을 바로 계산할 수 있는 준비 상태가 됨.

        # 1. 캐시 경로 설정 및 캐시 존재 여부 확인
        file_name = self._build_file_path(scenario)
        # metric_cache.pkl 파일을 저장할 전체 경로를 만듭니다.

        # 2. 이미 캐시가 존재하고 강제 재계산이 아니라면 캐시 로드
        assert file_name is not None, "Cache path can not be None for saving cache."
        if file_name.exists() and not self._force_feature_computation:
            return CacheMetadataEntry(file_name)
        # 이미 같은 시나리오에 대한 캐시 파일이 존재하고
        # force_feature_computation=False면
        # -> 캐시 재계산 없이 기존 파일 경로만 메타데이터로 반환
        # 📌 이걸 통해 중복 계산 방지 및 속도 최적화
        
        # 3. 캐시가 없거나 강제 재계산일 경우 → 새로 계산
        metric_cache = self.compute_metric_cache(scenario)
        # 이 한 줄이 실제 heavy한 작업의 핵심입니다. 내부적으로는:
        # 🧠 내부에서 수행되는 작업 (중요)
        #  - 플래너 입력 생성 → 시나리오 초기 상태 기반
        #  - PDMClosedPlanner 경로 생성 → 모델 기반 주행 시뮬레이션
        #  - GT 기반 객체 추적 정보 보간(interpolate) → 10Hz 정밀도 확보
        #  - 신호등 상태 보간
        #  - 관찰값 생성 (PDMObservation) → 경로 기반 상태 평가
        #  - Ego GT 궤적, 과거/미래 객체 궤적 추출
        #  - 맵 정보 설정
        #  - 최종적으로 MetricCache 객체 생성
        # 📌 MetricCache는 위 모든 데이터를 하나로 묶은 평가용 구조체입니다.

        # 4. 계산된 캐시를 디스크에 저장
        metric_cache.dump()
        # metric_cache.pkl 파일로 저장

        # 5. 캐시 메타데이터 객체 반환
        return CacheMetadataEntry(metric_cache.file_path)

    def _extract_ego_future_trajectory(self, scenario: NavSimScenario) -> Trajectory:
        ego_trajectory_sampling = TrajectorySampling(
            time_horizon=self._proposal_sampling.time_horizon,
            interval_length=scenario.database_interval,
        )
        future_ego_states = list(
            scenario.get_ego_future_trajectory(
                iteration=0,
                time_horizon=ego_trajectory_sampling.time_horizon,
                num_samples=ego_trajectory_sampling.num_poses,
            )
        )
        initial_ego_state = scenario.get_ego_state_at_iteration(0)
        if future_ego_states[0].time_point != initial_ego_state.time_point:
            # nuPlan does not return the initial state while navsim does
            # make sure to add the initial state before transforming to relative poses
            future_ego_states = [initial_ego_state] + future_ego_states

        future_ego_poses = [state.rear_axle for state in future_ego_states]
        relative_future_states = absolute_to_relative_poses(future_ego_poses)[1:]
        return Trajectory(
            poses=np.array([[pose.x, pose.y, pose.heading] for pose in relative_future_states]),
            trajectory_sampling=ego_trajectory_sampling,
        )

    def compute_metric_cache(self, scenario: NavSimScenario) -> MetricCache:
        # 이 함수는 NavSimScenario를 받아서 그 안의 주행 상황, 객체, 신호등, 지도 등을 기반으로
        # 정량화 가능한 시뮬레이션 캐시(MetricCache)를 생성함.
        # 1. 캐시 파일 경로 생성
        file_name = self._build_file_path(scenario)
        # 이 캐시 데이터를 어떤 경로에 저장할지 결정
        # 내부적으로 log_name/scenario_type/token/metric_cache.pkl 경로 구성

        # TODO: we should infer this from the scene metadata
        # 2. 합성 시나리오 여부 판단
        is_synthetic_scene = len(scenario.token) == 17
        # token 길이가 17이면 synthetic scene으로 판단 (현재는 간단한 heuristic 사용)
        # 이후 SceneFrameType을 결정할 때 사용됨

        # init and run PDM-Closed
        # 3. PDMClosedPlanner 초기화 + 예측 경로 생성
        planner_input, planner_initialization = self._get_planner_inputs(scenario)
        # → 시뮬레이션에 필요한 입력 준비
        # PlannerInitialization	지도 정보, 목적지, 경로 블록 등 초기 설정
        # PlannerInput	초기 ego 상태, 주변 객체 상태, 신호등 상태 등

        self._pdm_closed.initialize(planner_initialization)
        # → PDMClosedPlanner를 초기화합니다.
        # 이 플래너는 학습 기반이 아닌 IDM 기반 물리 시뮬레이터입니다.
        # 사용된 정책: BatchIDMPolicy
        # 예: 앞차와의 거리, 속도, 가속도 등을 기반으로 "안전하게 갈 수 있는 궤적" 계산

        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(planner_input)
        # → 실제로 궤적(trajectory)을 계산합니다.
        # 사용 방식:
        # 경로 중심선을 따라
        # 양옆 레인 변경을 시도하며
        # 속도 제한, 앞차 추종 등을 고려하여
        # 최적 또는 안전한 경로 후보를 생성

        # ❗ 이 궤적은 실제 모델 예측이 아닌, 시뮬레이션된 안전 기반 궤적입니다.

        # 평가용 플래너(PDMClosedPlanner)를 초기화하고 실행
        # 실제 주행 계획(Predicted trajectory)을 생성 (planning 관점에서 매우 중요)
        # 이 경로는 나중에 GT trajectory와 비교해 metric 평가 기준이 됨

        # 4. 경로 정보 로드
        route_roadblock_dict, route_lane_dict = self._load_route_dicts(
            scenario, planner_initialization.route_roadblock_ids
        )
        # 경로에 포함된 roadblock과 차선(lane)의 geometry를 불러옴
        # 이후 traffic light, 주변 객체 등과의 공간 관계 판단에 활용됨

        # 5. 객체 탐지 정보 보간 (10Hz 정밀도로)
        interpolated_detection_tracks = self._interpolate_gt_observation(scenario)
        # 주변 객체의 상태 (위치, 속도 등)를 시나리오 시간 단위(보통 2Hz) → 10Hz로 보간
        # StateInterpolator 사용
        # 이후 planner나 평가자가 시간 정렬된 고정밀 객체 정보를 활용할 수 있게 됨  

        # 6. 신호등 상태도 시간 단위로 보간
        interpolated_traffic_light_status = self._interpolate_traffic_light_status(scenario)
        # 프레임별 traffic light 상태를 시간 정렬된 리스트로 변환
        # 예: [초록 → 빨강 → 빨강 …] 리스트로 쭉 정리됨

        # 7. 관찰 정보 생성 (PDMObservation)
        observation = self._build_pdm_observation(
            interpolated_detection_tracks=interpolated_detection_tracks,
            interpolated_traffic_light_data=interpolated_traffic_light_status,
            route_lane_dict=route_lane_dict,
        )
        # 위에서 만든 객체/신호등/지도 정보를 기반으로 PDMObservation 객체 생성
        # planner가 판단한 경로와 어떤 상황에서 움직이는지를 평가에 활용할 수 있음

        # 8. 미래 객체 정보 준비
        future_tracked_objects = interpolated_detection_tracks[1:]
        # 현재 시점 이후 프레임에 해당하는 객체 상태만 분리해서 저장

        # 9. 과거 ego 궤적 생성
        past_human_trajectory = InterpolatedTrajectory(
            [ego_state for ego_state in scenario.get_ego_past_trajectory(0, 1.5)]
        )
        # ego 차량의 과거 상태를 시간 순서대로 모아 trajectory 구성
        # 시뮬레이터나 loss 계산 시 히스토리 기반 정보로 쓰임

        # 10. 미래 GT trajectory 구성 (합성 scene 제외)
        if not is_synthetic_scene:
            human_trajectory = self._extract_ego_future_trajectory(scenario)
        else:
            human_trajectory = None
        # 실제 로그 데이터는 미래 GT가 존재하므로 추출
        # 합성 데이터는 미래 GT가 없는 경우가 많아 생략

        # 11. MetricCache 객체 생성
        # save and dump features
        return MetricCache(
            file_path=file_name,
            log_name=scenario.log_name,
            scene_type=SceneFrameType.SYNTHETIC if is_synthetic_scene else SceneFrameType.ORIGINAL,
            timepoint=scenario.start_time,
            trajectory=pdm_closed_trajectory,
            human_trajectory=human_trajectory,
            past_human_trajectory=past_human_trajectory,
            ego_state=scenario.initial_ego_state,
            observation=observation,
            centerline=self._pdm_closed._centerline,
            route_lane_ids=list(self._pdm_closed._route_lane_dict.keys()),
            drivable_area_map=self._pdm_closed._drivable_area_map,
            past_detections_tracks=[
                dt for dt in scenario.get_past_tracked_objects(iteration=0, time_horizon=1.5, num_samples=3)
            ][:-1],
            current_tracked_objects=[scenario.initial_tracked_objects],
            future_tracked_objects=future_tracked_objects,
            map_parameters=MapParameters(
                map_root=scenario.map_root,
                map_version=scenario.map_version,
                map_name=scenario.map_api.map_name,
            ),
        )
        # ✅ 정리: 이 함수는 무엇을 하는가?
        # 단계	내용
        # 입력	하나의 NavSimScenario
        # 처리	플래너 실행, 객체/신호등 정보 보간, GT trajectory 생성
        # 출력	평가 기준 정보를 포함한 MetricCache 객체 (→ 캐싱됨)
