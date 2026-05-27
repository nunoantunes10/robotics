import csv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "src"))

from robot_state import RobotState
from controller import Supervisor
import numpy as np
import zmq


class RobotClient(Supervisor):
    TERMINATE_COMMAND = -1
    RESET_COMMAND = -2

    GOAL_POSITIONS = {
        0: (-4.25, -2.3),
        1: (-2.75, -2.3),
        2: (-1.26114, -2.31),
        3: (0.25, -2.3),
        4: (3.96, -0.7),
        5: (-1.74, 0.37),
        6: (-1.05, 0.37),
        7: (-2.55326, 1.63735),
        8: (-1.04326, 1.62735),
    }
    TRANSFER_FINETUNE_GOAL_POSITIONS = {
        0: (-4.5, -2.5),
        1: (0.0, -2.5),
        2: (4.5, -2.5),
        3: (-4.5, 1.7),
        4: (0.0, 1.7),
        5: (4.5, 1.7),
        6: (-4.5, 5.8),
        7: (0.0, 5.8),
        8: (4.5, 5.8),
    }
    REALWORLD_TEST_GOAL_POSITIONS = {
        0: (0.0, 7.2),
    }
    WORLD_GOAL_POSITIONS = {
        "smart-wheelchairs-transfer-finetune.wbt": TRANSFER_FINETUNE_GOAL_POSITIONS,
        "smart-wheelchairs-realworld-test.wbt": REALWORLD_TEST_GOAL_POSITIONS,
    }

    def __init__(self, id: int):
        super(RobotClient, self).__init__()

        self.id = id
        self.goal_positions = self.get_world_goal_positions()

        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        if sys.platform == "win32":
            port = 10000 + int(id)
            self.socket.connect(f"tcp://127.0.0.1:{port}")
        else:
            self.socket.connect(f"ipc:///tmp/giorgio_{id}")

        self.timestep = int(self.getBasicTimeStep())
        self.positions = []

        self.robot_node = self.getSelf()

        self.lidar = self.getDevice("Lidar")
        self.lidar.enable(self.timestep)

        self.bumper = self.getDevice("Bumper")
        self.bumper.enable(self.timestep)

        self.receiver = self.getDevice("Receiver")
        self.receiver.enable(self.timestep)

        self.emitter = self.getDevice("Emitter")

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        """ Distance between wheels """
        self.l = 0.12

        """ Store initial position for resetting """
        self.initial_position = self.robot_node.getField("translation").getSFVec3f()
        self.initial_rotation = self.robot_node.getField("rotation").getSFRotation()

        self.reset_robot()

    def reset_robot(self, rotate=True) -> None:
        """Resets the robot to its initial position."""
        self.robot_node.getField("translation").setSFVec3f(self.initial_position)

        if rotate:
            rotation = self.initial_rotation.copy()
            rotation[3] += np.random.uniform(-0.5, 0.5)  # Randomize rotation
            self.robot_node.getField("rotation").setSFRotation(rotation)

        self.simulationResetPhysics()

        # Seems to never have more than 1 packet in the queue
        while self.receiver.getQueueLength() > 0:
            self.receiver.nextPacket()

    def run(self) -> None:
        it = 0
        while self.step(self.timestep) != -1:
            start_pos = self.robot_node.getField("translation").getSFVec3f()
            self.positions.append(start_pos[:2])

            action = self.get_action()

            if self.is_command(action, self.TERMINATE_COMMAND):
                break
            if self.is_command(action, self.RESET_COMMAND):
                self.update_motors(np.array([0.0, 0.0], dtype=np.float32))
                self.reset_robot()
                self.positions = []
                self.send_observation(self.build_state(collided=False, goal_reached=False))
                continue
            if action.shape != (2,):
                self.send_observation(self.build_state(collided=False, goal_reached=False))
                continue

            self.update_motors(action)

            """Observation sent to server is lidar readings + collision/end flag"""
            collided = self.detect_collision()
            end = self.detect_end()

            if collided or end:
                log_dir = os.path.join(os.path.dirname(__file__), "../..", "logs")
                os.makedirs(log_dir, exist_ok=True)
                with open(
                    os.path.join(log_dir, f"positions_{self.id}_{it}.csv"),
                    "w",
                    newline="",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(["x", "y"])
                    writer.writerows(self.positions)

                self.positions = []
                it += 1
                self.reset_robot()

            state = self.build_state(collided=collided, goal_reached=end)
            self.send_observation(state)

        print("Simulation ended, saving trajectory...")

        print("Trajectory saved, resetting robot...")
        self.reset_robot(rotate=False)

    def get_action(self) -> np.ndarray:
        """Open pipe and read action from server"""
        return np.array(self.socket.recv_pyobj(), dtype=np.float32)

    def send_observation(self, obs: RobotState) -> None:
        """Open pipe and send observation to server"""
        self.socket.send_pyobj(obs)

    @staticmethod
    def is_command(action: np.ndarray, command: int) -> bool:
        return action.shape == (1,) and int(action[0]) == command

    def build_state(self, collided: bool, goal_reached: bool) -> RobotState:
        lidar = self.read_observation()
        pos = self.robot_node.getField("translation").getSFVec3f()
        rotation = self.robot_node.getField("rotation").getSFRotation()
        goal_distance, goal_bearing_sin, goal_bearing_cos = self.get_goal_features(
            pos,
            rotation,
        )

        return RobotState(
            lidar=lidar,
            prev_action=0,  # placeholder, will be set in env
            collided=collided,
            goal_reached=goal_reached,
            goal_distance=goal_distance,
            goal_bearing_sin=goal_bearing_sin,
            goal_bearing_cos=goal_bearing_cos,
        )

    def update_motors(self, action: np.ndarray) -> None:
        """
        Action is pair linear velocity, angular velocity
        Convert to left and right wheel speeds
        """
        left_speed = action[0] - action[1] * self.l / 2
        right_speed = action[0] + action[1] * self.l / 2
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def get_goal_distance(self, position) -> float | None:
        distance, _, _ = self.get_goal_features(
            position,
            self.robot_node.getField("rotation").getSFRotation(),
        )
        return distance

    def get_goal_features(self, position, rotation) -> tuple[float | None, float, float]:
        goal = self.goal_positions.get(self.id)
        if goal is None:
            return None, 0.0, 1.0

        dx = goal[0] - position[0]
        dy = goal[1] - position[1]
        goal_distance = float(np.hypot(dx, dy))
        target_yaw = float(np.arctan2(dy, dx))
        robot_yaw = self.yaw_from_rotation(rotation)
        goal_bearing = self.normalize_angle(target_yaw - robot_yaw)
        return goal_distance, float(np.sin(goal_bearing)), float(np.cos(goal_bearing))

    def get_world_goal_positions(self) -> dict[int, tuple[float, float]]:
        try:
            world_name = os.path.basename(self.getWorldPath())
        except Exception:
            world_name = ""
        return self.WORLD_GOAL_POSITIONS.get(world_name, self.GOAL_POSITIONS)

    @staticmethod
    def yaw_from_rotation(rotation) -> float:
        if len(rotation) < 4:
            return 0.0
        return float(rotation[2] * rotation[3])

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def read_observation(self) -> np.ndarray:
        """Clip to avoid inf or nan values"""
        try:
            image = self.lidar.getRangeImage()
        except ValueError:
            # Handle Webots ValueError: NULL pointer access before first update
            image = None

        if not image:
            res = self.lidar.getHorizontalResolution()
            layers = self.lidar.getNumberOfLayers()
            image = [10.0] * (res * layers)
            
        return np.clip(np.array(image), 0, 10)

    def detect_collision(self) -> bool:
        """Bumper value is 1 if collision is detected, else 0"""
        collided = self.bumper.getValue() == 1

        if collided:
            message = "collision".encode("utf-8")
            self.emitter.send(message)

        return collided

    def detect_end(self) -> bool:
        """
        End strip has an emmitter that sends a message when collision is detected
        Note that in the Webots world, the end strip sends messages in the same channel that the robot is listening
        And, of course, robots and strips on different corridors use different channels
        """

        return self.receiver.getQueueLength() > 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python robot_client.py <robot_id>")
        sys.exit(1)

    client = RobotClient(id=int(sys.argv[1]))
    client.run()
