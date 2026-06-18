from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    lidar: np.ndarray
    prev_action: int
    collided: bool = False
    goal_reached: bool = False
    position: np.ndarray | None = None
    yaw: float | None = None

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.lidar, [self.prev_action]])
