import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Graph",
                "normalize": "True",
                "absolute": False
            },
            "action": {
                "type": "ContinuousAction",
                # "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 20,
            "vehicles_density": 2,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 80,  # [s]
            "ego_spacing": 1.5,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0,    # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": -0.2,   # The reward received at each lane change action.
            "goal_diffrence_reward": -1,
            "acceleration reward": 0,
            "reward_speed_range": [15, 30],
            "offroad_terminal": True,
            "disable_collision_checks": True
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            # controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
            #     self.road,
            #     lane_index={'0','1',1},
            #     longitudinal=0.5,
            #     speed=25
            # )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                self.road.vehicles.append(
                    other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                )

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # # lane_changed = 0            # TODO: comment when not using discrete action
        # # if action == 0 or action == 2:        # TODO: comment when not using discrete action
        # #     lane_changed = 1        # TODO: comment when not using discrete action
        # lane_changed = action == 0 or action == 2
        # action_up = action == 3
        # action_down = action == 4
        # if self.vehicle.speed < 10:
        #     reward = 0
        # else:
        #     reward = \
        #         + self.config["collision_reward"] * self.vehicle.crashed \
        #         + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
        #         + self.config["accelaration reward"] * action_up \
        #         - self.config["acceleration reward"] * action_down * 2 \
        #         + self.config["lane_change_reward"] * lane_changed \
        #         # + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        #     reward = utils.lmap(reward,
        #                       [self.config["collision_reward"] + self.config["lane_change_reward"] - self.config["acceleration reward"] * 2,
        #                        self.config["high_speed_reward"] + self.config["acceleration reward"]],
        #                       [0, 1])
        # reward = 0 if not self.vehicle.on_road else reward
        # return reward

        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        action_speed = action[0]
        steering = abs(action[1])
        reward = self.config["collision_reward"] * self.vehicle.crashed \
                    + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
                    + self.config["acceleration reward"] * action_speed \
                    + self.config["lane_change_reward"] * steering
        reward = -1 if self.vehicle.speed < 20 else reward
        reward = utils.lmap(reward,
                    [self.config["collision_reward"] - self.config["acceleration reward"]
                        + self.config["lane_change_reward"],
                     self.config["high_speed_reward"] + self.config["acceleration reward"]],
                    [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)
