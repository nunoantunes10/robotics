from robot_state import RobotState
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import zmq
import sys


ACTION_FORWARD = 0
ACTION_FORWARD_LEFT = 1
ACTION_FORWARD_RIGHT = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_STOP = 5

STRAIGHT_BONUS_CLEAR = 0.25
SOFT_FORWARD_BONUS_CLEAR = 0.08
TURN_PENALTY_CLEAR = 0.15
STOP_PENALTY_CLEAR = 0.10
OSCILLATION_PENALTY = 0.20
STRONG_OSCILLATION_PENALTY = 0.40
NON_FORWARD_CLEAR_PENALTY = 0.30
CLEAR_FRONT_THRESHOLD = 1.5
DANGER_FRONT_THRESHOLD = 0.75
GOAL_DIRECTION_PROGRESS_WEIGHT = 2.0
GOAL_DIRECTION_PROGRESS_CLIP = 0.5
GOAL_DIRECTION_BACKWARD_PENALTY = 0.10
GOAL_DIRECTION_LOOP_PENALTY = 0.05
GOAL_DIRECTION_LOOP_WINDOW = 20
GOAL_DIRECTION_MIN_RECENT_PROGRESS = 0.02
GOAL_DIRECTION_CLEAR_FRONT_THRESHOLD = 1.5
GOAL_DIRECTION_HEADING_WEIGHT = 0.05


class WheelchairEnv(gym.Env):
    def __init__(
        self,
        env_id: int,
        lidar_dim: int = 360,
        reward_mode: str = "default",
        straight_bonus_clear: float = STRAIGHT_BONUS_CLEAR,
        soft_forward_bonus_clear: float = SOFT_FORWARD_BONUS_CLEAR,
        turn_penalty_clear: float = TURN_PENALTY_CLEAR,
        stop_penalty_clear: float = STOP_PENALTY_CLEAR,
        oscillation_penalty: float = OSCILLATION_PENALTY,
        strong_oscillation_penalty: float = STRONG_OSCILLATION_PENALTY,
        non_forward_clear_penalty: float = NON_FORWARD_CLEAR_PENALTY,
        clear_front_threshold: float = CLEAR_FRONT_THRESHOLD,
        danger_front_threshold: float = DANGER_FRONT_THRESHOLD,
        goal_direction_progress_weight: float = GOAL_DIRECTION_PROGRESS_WEIGHT,
        goal_direction_progress_clip: float = GOAL_DIRECTION_PROGRESS_CLIP,
        goal_direction_backward_penalty: float = GOAL_DIRECTION_BACKWARD_PENALTY,
        goal_direction_loop_penalty: float = GOAL_DIRECTION_LOOP_PENALTY,
        goal_direction_loop_window: int = GOAL_DIRECTION_LOOP_WINDOW,
        goal_direction_min_recent_progress: float = GOAL_DIRECTION_MIN_RECENT_PROGRESS,
        goal_direction_clear_front_threshold: float = (
            GOAL_DIRECTION_CLEAR_FRONT_THRESHOLD
        ),
        goal_direction_heading_weight: float = GOAL_DIRECTION_HEADING_WEIGHT,
    ):
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
        self.reward_mode = reward_mode
        self.straight_bonus_clear = straight_bonus_clear
        self.soft_forward_bonus_clear = soft_forward_bonus_clear
        self.turn_penalty_clear = turn_penalty_clear
        self.stop_penalty_clear = stop_penalty_clear
        self.oscillation_penalty = oscillation_penalty
        self.strong_oscillation_penalty = strong_oscillation_penalty
        self.non_forward_clear_penalty = non_forward_clear_penalty
        self.clear_front_threshold = clear_front_threshold
        self.danger_front_threshold = danger_front_threshold
        self.goal_direction_progress_weight = goal_direction_progress_weight
        self.goal_direction_progress_clip = goal_direction_progress_clip
        self.goal_direction_backward_penalty = goal_direction_backward_penalty
        self.goal_direction_loop_penalty = goal_direction_loop_penalty
        self.goal_direction_loop_window = goal_direction_loop_window
        self.goal_direction_min_recent_progress = goal_direction_min_recent_progress
        self.goal_direction_clear_front_threshold = goal_direction_clear_front_threshold
        self.goal_direction_heading_weight = goal_direction_heading_weight
        self.last_actions = []
        self.reset_goal_direction_state()

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

        action_id = int(action)
        self.prev_action = action_id
        motor_action = self.to_action(action_id)
        reward = self.get_reward(self.prev_lidar, motor_action)
        self.record_action(action_id)
        transfer_reward, reward_debug = self.transfer_finetune_reward(
            self.prev_lidar, action_id
        )
        reward += transfer_reward

        obs = self.send_action_get_obs(motor_action)
        goal_direction_reward, goal_direction_debug = self.goal_direction_reward(obs)
        reward += goal_direction_reward

        if obs.collided:
            reward += self.collision_reward()
        elif obs.goal_reached:
            reward += self.goal_reward()

        self.prev_lidar = obs.lidar
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
            "reward_mode": self.reward_mode,
        }
        info.update(reward_debug)
        info.update(goal_direction_debug)

        return self.state_to_array(obs), reward, done, False, info

    def get_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        v, w = action


        r_distance = 1.0 if v > 0 else 0.0

        # Collision penalty (exponential when very close)
        min_range = np.min(obs[140:220])  # Front sector
        collision_threshold = 1.0
        if min_range < collision_threshold:
            r_collision = -np.exp(3 * (collision_threshold - min_range)) + 1
        else:
            r_collision = 0.0

        # Main navigation reward - replaces both direction_reward and early_side_commitment
        r_navigation = self.navigation_reward(obs, action)

        # Penalize excessive turning when not needed
        r_stability = self.stability_reward(obs, action)

        total_reward = r_distance + r_collision + r_navigation + r_stability
        return total_reward

    def record_action(self, action_id: int) -> None:
        self.last_actions.append(action_id)
        self.last_actions = self.last_actions[-6:]

    def transfer_finetune_reward(
        self, obs: np.ndarray, action_id: int
    ) -> Tuple[float, dict]:
        front_sector = obs[170:190]
        front_clearance = float(np.min(front_sector)) if len(front_sector) else 0.0
        debug = {
            "front_clearance": front_clearance,
            "straight_reward_bonus": 0.0,
            "turn_penalty": 0.0,
            "stop_penalty": 0.0,
            "oscillation_penalty": 0.0,
            "non_forward_penalty": 0.0,
            "transfer_reward": 0.0,
        }

        if self.reward_mode != "transfer_finetune":
            return 0.0, debug

        if front_clearance < self.danger_front_threshold:
            return 0.0, debug

        reward = 0.0
        if front_clearance > self.clear_front_threshold:
            if action_id == ACTION_FORWARD:
                debug["straight_reward_bonus"] = self.straight_bonus_clear
                reward += debug["straight_reward_bonus"]
            elif action_id in (ACTION_FORWARD_LEFT, ACTION_FORWARD_RIGHT):
                debug["straight_reward_bonus"] = self.soft_forward_bonus_clear
                reward += debug["straight_reward_bonus"]
            elif action_id in (ACTION_LEFT, ACTION_RIGHT):
                debug["turn_penalty"] = -self.turn_penalty_clear
                reward += debug["turn_penalty"]
            elif action_id == ACTION_STOP:
                debug["stop_penalty"] = -self.stop_penalty_clear
                reward += debug["stop_penalty"]

            recent = self.last_actions[-5:]
            forward_count = sum(a == ACTION_FORWARD for a in recent)
            if len(recent) == 5 and forward_count <= 1:
                debug["non_forward_penalty"] = -self.non_forward_clear_penalty
                reward += debug["non_forward_penalty"]

        if len(self.last_actions) >= 4:
            recent = self.last_actions[-4:]
            turnish_actions = {
                ACTION_LEFT,
                ACTION_RIGHT,
                ACTION_FORWARD_LEFT,
                ACTION_FORWARD_RIGHT,
            }
            if all(a in turnish_actions for a in recent):
                debug["oscillation_penalty"] = -self.oscillation_penalty
                reward += debug["oscillation_penalty"]

            if recent in (
                [ACTION_LEFT, ACTION_RIGHT, ACTION_LEFT, ACTION_RIGHT],
                [ACTION_RIGHT, ACTION_LEFT, ACTION_RIGHT, ACTION_LEFT],
            ):
                debug["oscillation_penalty"] -= self.strong_oscillation_penalty
                reward -= self.strong_oscillation_penalty

        debug["transfer_reward"] = reward
        return reward, debug

    def reset_goal_direction_state(self):
        self.initial_position = None
        self.previous_position = None
        self.initial_forward_vector = None
        self.previous_forward_progress = 0.0
        self.initial_yaw = None
        self.previous_yaw = None
        self.recent_forward_progress = []

    def goal_direction_debug(self, obs: RobotState = None):
        front_clearance = None
        if obs is not None and obs.lidar is not None:
            front_sector = obs.lidar[170:190]
            front_clearance = (
                float(np.min(front_sector)) if len(front_sector) else 0.0
            )

        return {
            "pose_available": False,
            "goal_direction_progress": 0.0,
            "goal_direction_progress_delta": 0.0,
            "goal_direction_progress_reward": 0.0,
            "goal_direction_backward_penalty": 0.0,
            "goal_direction_heading_error": None,
            "goal_direction_heading_penalty": 0.0,
            "goal_direction_loop_penalty": 0.0,
            "goal_direction_reward": 0.0,
            "front_clearance": front_clearance,
        }

    def goal_direction_reward(self, state: RobotState) -> Tuple[float, dict]:
        if self.reward_mode != "transfer_goal_direction":
            return 0.0, {}

        debug = self.goal_direction_debug(state)
        if not self.initialize_goal_direction_state(state):
            return 0.0, debug

        current_position = self._get_robot_position_2d(state)
        current_yaw = self._get_robot_yaw(state)
        if current_position is None:
            return 0.0, debug

        displacement = current_position - self.initial_position
        forward_progress = float(np.dot(displacement, self.initial_forward_vector))
        progress_delta = forward_progress - self.previous_forward_progress
        self.previous_forward_progress = forward_progress
        self.previous_position = current_position
        self.previous_yaw = current_yaw

        progress_reward = float(
            np.clip(
                self.goal_direction_progress_weight * progress_delta,
                -self.goal_direction_progress_clip,
                self.goal_direction_progress_clip,
            )
        )

        backward_penalty = 0.0
        if progress_delta < -0.01:
            backward_penalty = -self.goal_direction_backward_penalty

        heading_error = None
        heading_penalty = 0.0
        front_clearance = debug["front_clearance"]
        if (
            current_yaw is not None
            and front_clearance is not None
            and front_clearance > self.goal_direction_clear_front_threshold
        ):
            heading_error = abs(self._angle_diff(current_yaw, self.initial_yaw))
            heading_penalty = -self.goal_direction_heading_weight * heading_error

        self.recent_forward_progress.append(forward_progress)
        self.recent_forward_progress = self.recent_forward_progress[
            -self.goal_direction_loop_window:
        ]
        loop_penalty = 0.0
        if len(self.recent_forward_progress) == self.goal_direction_loop_window:
            net_recent_progress = (
                self.recent_forward_progress[-1] - self.recent_forward_progress[0]
            )
            if net_recent_progress < self.goal_direction_min_recent_progress:
                loop_penalty = -self.goal_direction_loop_penalty

        total_reward = progress_reward + backward_penalty + heading_penalty + loop_penalty
        debug.update({
            "pose_available": True,
            "goal_direction_progress": forward_progress,
            "goal_direction_progress_delta": progress_delta,
            "goal_direction_progress_reward": progress_reward,
            "goal_direction_backward_penalty": backward_penalty,
            "goal_direction_heading_error": (
                float(heading_error) if heading_error is not None else None
            ),
            "goal_direction_heading_penalty": float(heading_penalty),
            "goal_direction_loop_penalty": float(loop_penalty),
            "goal_direction_reward": float(total_reward),
        })
        return total_reward, debug

    def initialize_goal_direction_state(self, state: RobotState) -> bool:
        if self.initial_position is not None and self.initial_forward_vector is not None:
            return True

        position = self._get_robot_position_2d(state)
        yaw = self._get_robot_yaw(state)
        if position is None or yaw is None:
            return False

        forward_vector = self._normalize_vector(
            np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        )
        if np.linalg.norm(forward_vector) == 0:
            return False

        self.initial_position = position
        self.previous_position = position
        self.initial_forward_vector = forward_vector
        self.previous_forward_progress = 0.0
        self.initial_yaw = yaw
        self.previous_yaw = yaw
        self.recent_forward_progress = [0.0]
        return True

    def _get_robot_position_2d(self, state: RobotState):
        if state.position is None:
            return None
        position = np.asarray(state.position, dtype=np.float64)
        if position.shape[0] < 2:
            return None
        return position[:2]

    def _get_robot_yaw(self, state: RobotState):
        if state.yaw is None:
            return None
        return float(state.yaw)

    def _angle_diff(self, a: float, b: float) -> float:
        return float(np.arctan2(np.sin(a - b), np.cos(a - b)))

    def _normalize_vector(self, vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < eps:
            return vector
        return vector / norm

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
        _, w = action

        # Light penalty for turning (encourages smoother paths)
        if w != 0:
            return -0.2
        return 0

    def reset_preference(self):
        self.prev_pref = 0.0

    def collision_reward(self) -> int:
        return -20

    def goal_reward(self) -> int:
        return 0

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
        self.last_actions = []
        self.reset_goal_direction_state()
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
            "reward_mode": self.reward_mode,
        }

        return obs, info

    def close(self):
        print("Closing environment " + str(self.env_id))
        self.socket.send_pyobj([-1])
        self.socket.close()
        zmq.Context.instance().destroy()
        return super().close()
