import numpy as np
import torch

from highway_env.envs.highway_env import HighwayEnv
from highway_env import utils
from rlkit.envs import register_env     # only used with rlkit
# from highway_env.envs import register_env
# from gym.envs.registration import register
#
# register(id='highwayspeed-v0',
#          entry_point='highway_env.envs.highway_meta:HighwayMetaEnv')

@register_env('highway-speed')
class HighwayMetaEnv(HighwayEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        super(HighwayMetaEnv, self).__init__()

    def _reward(self, action, goal_speed) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        action_speed = action[0]
        steering = abs(action[1])
        # speed_difference = abs(self.vehicle.speed - goal_speed) / 10
        speed_difference = utils.lmap(abs(self.vehicle.speed - goal_speed), [0, 15], [0, 1])
        reward = self.config["collision_reward"] * self.vehicle.crashed \
                    + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
                    + self.config["acceleration reward"] * action_speed \
                    + self.config["lane_change_reward"] * steering \
                    + self.config["goal_diffrence_reward"] * speed_difference
        # reward = -1 if self.vehicle.speed < 20 else reward
        reward = utils.lmap(reward,
                    [self.config["collision_reward"] - self.config["acceleration reward"]
                        + self.config["lane_change_reward"] + self.config["goal_diffrence_reward"],
                     self.config["high_speed_reward"] + self.config["acceleration reward"]],
                    [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def step(self, action):
        """
        Perform an action and step the environment dynamics.
        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action, self._goal)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info

    def _info(self, obs, action):
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
            info["goal"] = self._task
        except NotImplementedError:
            pass
        return info

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        # velocities = np.random.uniform(15, 30, size=(num_tasks,))
        velocities = [30 for _ in range(10)]   #only for defined tasks
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks