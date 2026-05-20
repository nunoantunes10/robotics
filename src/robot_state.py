from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    lidar: np.ndarray
    prev_action: int
    collided: bool = False
    goal_reached: bool = False
    goal_distance: float | None = None
    goal_bearing_sin: float = 0.0
    goal_bearing_cos: float = 1.0

    def to_array(self) -> np.ndarray:
        goal_distance = 1.0 if self.goal_distance is None else self.goal_distance / 10.0
        lidar = np.nan_to_num(self.lidar / 10.0, nan=1.0, posinf=1.0, neginf=0.0)
        goal_bearing_sin = np.nan_to_num(self.goal_bearing_sin, nan=0.0)
        goal_bearing_cos = np.nan_to_num(self.goal_bearing_cos, nan=1.0)
        prev_action = np.zeros(6, dtype=np.float32)
        if 0 <= int(self.prev_action) < prev_action.size:
            prev_action[int(self.prev_action)] = 1.0
        return np.concatenate(
            [
                np.clip(lidar, 0.0, 1.0),
                [
                    np.clip(goal_distance, 0.0, 1.0),
                    np.clip(goal_bearing_sin, -1.0, 1.0),
                    np.clip(goal_bearing_cos, -1.0, 1.0),
                ],
                prev_action,
            ]
        ).astype(np.float32)
