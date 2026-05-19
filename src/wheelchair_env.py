from robot_state import RobotState
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import zmq
import sys


class WheelchairEnv(gym.Env):
    def __init__(self, env_id: int, lidar_dim: int = 360):
        super(WheelchairEnv, self).__init__()

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        if sys.platform == "win32":
            port = 10000 + int(env_id)
            self.socket.bind(f"tcp://127.0.0.1:{port}")
        else:
            self.socket.bind(f"ipc:///tmp/giorgio_{env_id}")

        self.env_id = env_id
        self.full_lidar_dim = 360
        self.lidar_dim = lidar_dim

        """
        Action and state space definition.
        Robot will be able to control speed of left and right wheels between 0 and 5.
        State is a vector of lidar readings and the last action taken (as integer).
        """
        self.action_space = gym.spaces.Discrete(6)

        v, w = 1, 3
        self.to_action = lambda x: (
            [
                [v, 0],   # Forward
                [v, w],   # Forward and left
                [v, -w],  # Forward and right
                [0, w],   # Left
                [0, -w],  # Right
                [0, 0],   # Stop
            ]
        )[x]

        self.observation_space = Box(
            low=np.concatenate([np.full(self.lidar_dim, 0.0), [0]]),
            high=np.concatenate([np.full(self.lidar_dim, 10.0), [5]]),
            dtype=np.float64,
        )
        self.obs_shape = self.observation_space.shape

        self.no_obs()
        self.prev_action = 0
        self.prev_pref = 0.0
        self.prev_goal_distance = None
        self.time_step = 0
        self.time_limit = 20_000
        self.commitment_threshold = 2.5

    def downsample_lidar(self, lidar: np.ndarray) -> np.ndarray:
        if len(lidar) == self.lidar_dim:
            return lidar.astype(np.float64)

        indices = np.linspace(0, len(lidar) - 1, self.lidar_dim, dtype=int)
        return lidar[indices].astype(np.float64)

    def state_to_array(self, state: RobotState) -> np.ndarray:
        lidar = self.downsample_lidar(state.lidar)
        return np.concatenate([lidar, [state.prev_action]]).astype(np.float64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Takes an action and returns the next observation, reward, done flag, and info.
        :param action: Action to be taken (int)
        :return: Tuple of (observation, reward, done, info)
        """

        self.prev_action = action
        action = self.to_action(action)
        obs = self.send_action_get_obs(action)
        reward, reward_info = self.get_reward(obs, action)

        self.prev_lidar = obs.lidar
        self.prev_goal_distance = obs.goal_distance
        self.time_step += 1

        done = obs.collided or obs.goal_reached
        info = {
            "is_success": obs.goal_reached,
            "collision": obs.collided,
            "goal_reached": obs.goal_reached,
            "time_step": self.time_step,
            "reward": reward,
            "env_id": self.env_id,
            "lidar_dim": self.lidar_dim,
            "goal_distance": obs.goal_distance,
            "progress": reward_info["progress"],
            "min_front": reward_info["min_front"],
        }

        return self.state_to_array(obs), reward, done, False, info

    def get_reward(self, obs: RobotState, action: Tuple[int, int]) -> Tuple[float, dict]:
        v, w = action
        reward_info = {
            "progress": None,
            "min_front": self.get_min_front(obs.lidar),
        }

        if obs.collided:
            return self.collision_reward(), reward_info
        if obs.goal_reached:
            return self.goal_reward(), reward_info

        reward = -0.01

        if self.prev_goal_distance is not None and obs.goal_distance is not None:
            progress = self.prev_goal_distance - obs.goal_distance
            reward_info["progress"] = progress
            reward += 10.0 * progress

        if v > 0:
            reward += 0.05
        else:
            reward -= 0.02

        # Danger penalties use post-action LiDAR so near-obstacle states are not rewarded.
        min_front = reward_info["min_front"]
        if min_front is not None:
            if min_front < 1.0:
                reward -= 0.25
            if min_front < 0.5:
                reward -= 1.0

        return float(reward), reward_info

    @staticmethod
    def get_min_front(lidar: np.ndarray) -> float | None:
        front_sector = lidar[170:190]
        if front_sector.size == 0:
            return None
        return float(np.min(front_sector))

    def navigation_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        _, w = action

        # Define sectors on full 360 LiDAR
        left_sector = obs[100:170]
        front_sector = obs[170:190]
        right_sector = obs[190:260]

        # Calculate clearances
        left_clearance = np.mean(left_sector)
        right_clearance = np.mean(right_sector)

        # Check if there's an obstacle ahead that requires decision
        obstacle_ahead = np.any(front_sector < self.commitment_threshold)

        if obstacle_ahead:
            clearance_diff = right_clearance - left_clearance

            alpha = 0.40
            self.prev_pref = (1 - alpha) * self.prev_pref + alpha * clearance_diff

            r = 3.0

            if self.prev_pref > 0.2:
                return r if w < 0 else -r
            if self.prev_pref < -0.2:
                return r if w > 0 else -r

            if clearance_diff > 0.2:
                return r if w < 0 else -r * 0.5
            if clearance_diff < -0.2:
                return r if w > 0 else -r * 0.5
            if np.min(front_sector) < 1.0:
                return r if w != 0 else -r
        elif w == 0:
            return 1.0

        return 0

    def stability_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        return 0

    def reset_preference(self):
        self.prev_pref = 0.0

    def collision_reward(self) -> float:
        return -500.0

    def goal_reward(self) -> float:
        return 1000.0

    def send_action_get_obs(self, action: Tuple[int, int]) -> RobotState:
        self.socket.send_pyobj(action)
        return self.get_observation()

    def get_observation(self) -> RobotState:
        state = self.socket.recv_pyobj()
        state.prev_action = self.prev_action
        return state

    def no_obs(self) -> np.ndarray:
        return np.zeros(self.obs_shape, dtype=np.float64)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, dict]:
        self.prev_action = 0
        self.time_step = 0
        self.prev_goal_distance = None
        self.reset_preference()

        obs = self.no_obs()
        self.prev_lidar = np.zeros(self.full_lidar_dim, dtype=np.float64)

        info = {
            "is_success": False,
            "collision": False,
            "goal_reached": False,
            "time_step": self.time_step,
            "reward": 0.0,
            "env_id": self.env_id,
            "lidar_dim": self.lidar_dim,
            "goal_distance": None,
            "progress": None,
            "min_front": None,
        }

        return obs, info

    def close(self):
        print("Closing environment " + str(self.env_id))
        self.socket.send_pyobj([-1])
        self.socket.close()
        zmq.Context.instance().destroy()
        return super().close()
