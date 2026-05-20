from robot_state import RobotState
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import zmq
import sys


class WheelchairEnv(gym.Env):
    TERMINATE_COMMAND = -1
    RESET_COMMAND = -2
    SOCKET_TIMEOUT_MS = 10_000
    STOP_ACTION = 5
    MAX_LIDAR_RANGE = 10.0
    MAX_GOAL_DISTANCE = 10.0
    PROGRESS_CLIP = 0.25
    TIMEOUT_REWARD = -25.0

    def __init__(self, env_id: int, lidar_dim: int = 360):
        super(WheelchairEnv, self).__init__()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.SOCKET_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, self.SOCKET_TIMEOUT_MS)
        self.waiting_for_reply = False
        self.last_request = None
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
        State is normalized LiDAR, goal direction/distance, and previous action one-hot.
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

        self.goal_context_dim = 3
        self.prev_action_dim = self.action_space.n
        low = np.concatenate(
            [
                np.zeros(self.lidar_dim, dtype=np.float32),
                np.array([0.0, -1.0, -1.0], dtype=np.float32),
                np.zeros(self.prev_action_dim, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.ones(self.lidar_dim, dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.ones(self.prev_action_dim, dtype=np.float32),
            ]
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.obs_shape = self.observation_space.shape

        self.no_obs()
        self.prev_action = self.STOP_ACTION
        self.prev_pref = 0.0
        self.prev_goal_distance = None
        self.cached_reset_obs = None
        self.last_state = None
        self.time_step = 0
        self.time_limit = 3_000
        self.commitment_threshold = 2.5

    def downsample_lidar(self, lidar: np.ndarray) -> np.ndarray:
        if len(lidar) == self.lidar_dim:
            return lidar.astype(np.float64)

        indices = np.linspace(0, len(lidar) - 1, self.lidar_dim, dtype=int)
        return lidar[indices].astype(np.float64)

    def state_to_array(self, state: RobotState) -> np.ndarray:
        lidar = np.nan_to_num(
            self.downsample_lidar(state.lidar) / self.MAX_LIDAR_RANGE,
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        lidar = np.clip(
            lidar,
            0.0,
            1.0,
        )
        goal_distance = self.normalize_goal_distance(state.goal_distance)
        goal_bearing_sin = np.nan_to_num(state.goal_bearing_sin, nan=0.0)
        goal_bearing_cos = np.nan_to_num(state.goal_bearing_cos, nan=1.0)
        goal_bearing_sin = np.clip(goal_bearing_sin, -1.0, 1.0)
        goal_bearing_cos = np.clip(goal_bearing_cos, -1.0, 1.0)
        prev_action = self.prev_action_one_hot(state.prev_action)

        return np.concatenate(
            [
                lidar,
                [goal_distance, goal_bearing_sin, goal_bearing_cos],
                prev_action,
            ]
        ).astype(np.float32)

    def normalize_goal_distance(self, goal_distance: float | None) -> float:
        if goal_distance is None:
            return 1.0
        goal_distance = np.nan_to_num(goal_distance, nan=self.MAX_GOAL_DISTANCE)
        return float(np.clip(goal_distance / self.MAX_GOAL_DISTANCE, 0.0, 1.0))

    def prev_action_one_hot(self, prev_action: int) -> np.ndarray:
        one_hot = np.zeros(self.prev_action_dim, dtype=np.float32)
        if 0 <= int(prev_action) < self.prev_action_dim:
            one_hot[int(prev_action)] = 1.0
        return one_hot

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes an action and returns the next observation, reward, done flag, and info.
        :param action: Action to be taken (int)
        :return: Tuple of (observation, reward, done, info)
        """

        self.prev_action = int(action)
        motor_action = self.to_action(self.prev_action)
        obs = self.send_action_get_obs(motor_action)
        reward, reward_info = self.get_reward(obs, motor_action)

        self.prev_lidar = obs.lidar
        self.prev_goal_distance = obs.goal_distance
        self.last_state = obs
        self.time_step += 1

        terminated = obs.collided or obs.goal_reached
        truncated = self.time_step >= self.time_limit and not terminated
        if truncated:
            reward += self.timeout_reward()
        if terminated:
            self.cached_reset_obs = obs
        info = {
            "is_success": obs.goal_reached,
            "collision": obs.collided,
            "goal_reached": obs.goal_reached,
            "time_limit_reached": truncated,
            "time_step": self.time_step,
            "reward": reward,
            "env_id": self.env_id,
            "lidar_dim": self.lidar_dim,
            "goal_distance": obs.goal_distance,
            "goal_bearing_sin": obs.goal_bearing_sin,
            "goal_bearing_cos": obs.goal_bearing_cos,
            "progress": reward_info["progress"],
            "raw_progress": reward_info["raw_progress"],
            "min_front": reward_info["min_front"],
            "left_clearance": reward_info["left_clearance"],
            "right_clearance": reward_info["right_clearance"],
        }

        return self.state_to_array(obs), reward, terminated, truncated, info

    def get_reward(self, obs: RobotState, action: Tuple[int, int]) -> Tuple[float, dict]:
        v, w = action
        reward_info = {
            "progress": None,
            "raw_progress": None,
            "min_front": self.get_min_front(obs.lidar),
            "left_clearance": None,
            "right_clearance": None,
        }
        left_clearance, right_clearance = self.get_side_clearance(obs.lidar)
        reward_info["left_clearance"] = left_clearance
        reward_info["right_clearance"] = right_clearance

        if obs.collided:
            return self.collision_reward(), reward_info
        if obs.goal_reached:
            return self.goal_reward(), reward_info

        reward = -0.005
        min_front = reward_info["min_front"]
        progress = None

        if self.prev_goal_distance is not None and obs.goal_distance is not None:
            raw_progress = self.prev_goal_distance - obs.goal_distance
            progress = float(
                np.clip(raw_progress, -self.PROGRESS_CLIP, self.PROGRESS_CLIP)
            )
            reward_info["raw_progress"] = raw_progress
            reward_info["progress"] = progress
            progress_scale = 6.0
            if min_front is not None and min_front < 0.6:
                progress_scale = 2.0
            reward += progress_scale * progress

        if v > 0:
            if min_front is None or min_front >= 1.0:
                reward += 0.03
            else:
                reward -= 0.25
            if w != 0 and progress is not None and progress > 0:
                reward += 0.02
        else:
            if min_front is not None and min_front < 0.8 and w != 0:
                reward += 0.02
            else:
                reward -= 0.03

        # Danger penalties use post-action LiDAR so near-obstacle states are not rewarded.
        if min_front is not None:
            if min_front < 0.8:
                reward -= 0.25
            if min_front < 0.5:
                reward -= 0.75
            if min_front < 0.3:
                reward -= 1.50

        return float(reward), reward_info

    @staticmethod
    def get_min_front(lidar: np.ndarray) -> float | None:
        front_sector = lidar[170:190]
        if front_sector.size == 0:
            return None
        return float(np.min(front_sector))

    @staticmethod
    def get_side_clearance(lidar: np.ndarray) -> Tuple[float | None, float | None]:
        left_sector = lidar[100:170]
        right_sector = lidar[190:260]
        left_clearance = float(np.mean(left_sector)) if left_sector.size else None
        right_clearance = float(np.mean(right_sector)) if right_sector.size else None
        return left_clearance, right_clearance

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
        return -50.0

    def goal_reward(self) -> float:
        return 100.0

    def timeout_reward(self) -> float:
        return self.TIMEOUT_REWARD

    def send_action_get_obs(self, action: Tuple[int, int]) -> RobotState:
        self.send_request(action)
        return self.get_observation()

    def send_reset_get_obs(self) -> RobotState:
        self.send_request([self.RESET_COMMAND])
        return self.get_observation()

    def send_request(self, request) -> None:
        if self.waiting_for_reply:
            raise RuntimeError(
                f"Env {self.env_id} cannot send {request}; still waiting for "
                f"reply to {self.last_request}"
            )
        try:
            self.socket.send_pyobj(request)
        except zmq.Again as exc:
            raise TimeoutError(
                f"Env {self.env_id} timed out sending request {request}"
            ) from exc
        except zmq.ZMQError as exc:
            raise RuntimeError(
                f"Env {self.env_id} failed sending request {request}: {exc}"
            ) from exc
        self.waiting_for_reply = True
        self.last_request = request

    def get_observation(self) -> RobotState:
        try:
            state = self.socket.recv_pyobj()
        except zmq.Again as exc:
            raise TimeoutError(
                f"Env {self.env_id} timed out waiting for reply to "
                f"{self.last_request}. Check robot_client controller {self.env_id}."
            ) from exc
        except zmq.ZMQError as exc:
            raise RuntimeError(
                f"Env {self.env_id} failed receiving reply to "
                f"{self.last_request}: {exc}"
            ) from exc
        self.waiting_for_reply = False
        state.prev_action = self.prev_action
        return state

    def no_obs(self) -> np.ndarray:
        return np.zeros(self.obs_shape, dtype=np.float32)

    def reset(self, seed: int = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.prev_action = self.STOP_ACTION
        self.time_step = 0
        self.prev_goal_distance = None
        self.reset_preference()

        if self.cached_reset_obs is not None:
            obs = self.cached_reset_obs
            self.cached_reset_obs = None
        elif self.last_state is not None:
            obs = self.last_state
        else:
            obs = None

        if obs is not None:
            obs.collided = False
            obs.goal_reached = False
            self.prev_lidar = obs.lidar
            self.prev_goal_distance = obs.goal_distance
            min_front = self.get_min_front(obs.lidar)
            left_clearance, right_clearance = self.get_side_clearance(obs.lidar)
            reset_obs = self.state_to_array(obs)
            goal_distance = obs.goal_distance
            goal_bearing_sin = obs.goal_bearing_sin
            goal_bearing_cos = obs.goal_bearing_cos
        else:
            self.prev_lidar = np.zeros(self.full_lidar_dim, dtype=np.float64)
            self.prev_goal_distance = None
            min_front = None
            left_clearance = None
            right_clearance = None
            reset_obs = self.no_obs()
            goal_distance = None
            goal_bearing_sin = 0.0
            goal_bearing_cos = 1.0

        info = {
            "is_success": False,
            "collision": False,
            "goal_reached": False,
            "time_limit_reached": False,
            "time_step": self.time_step,
            "reward": 0.0,
            "env_id": self.env_id,
            "lidar_dim": self.lidar_dim,
            "goal_distance": goal_distance,
            "goal_bearing_sin": goal_bearing_sin,
            "goal_bearing_cos": goal_bearing_cos,
            "progress": None,
            "raw_progress": None,
            "min_front": min_front,
            "left_clearance": left_clearance,
            "right_clearance": right_clearance,
        }

        return reset_obs, info

    def close(self):
        print("Closing environment " + str(self.env_id))
        if not self.waiting_for_reply:
            try:
                self.send_request([self.TERMINATE_COMMAND])
            except Exception as exc:
                print(f"Could not send terminate to env {self.env_id}: {exc}")
        else:
            print(
                f"Env {self.env_id} is waiting for reply to {self.last_request}; "
                "closing socket without terminate command"
            )
        self.socket.close(linger=0)
        self.context.term()
        return super().close()
