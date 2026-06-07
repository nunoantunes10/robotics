import csv
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "src"))

from robot_state import RobotState
from controller import Supervisor
import numpy as np
import sys
import zmq


class RobotClient(Supervisor):
    def __init__(self, id: int):
        super(RobotClient, self).__init__()

        self.id = id

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

    def save_positions(self, episode_id: int) -> None:
        if not self.positions:
            return

        log_dir = os.path.join(os.path.dirname(__file__), "../..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(
            os.path.join(log_dir, f"positions_{self.id}_{episode_id}.csv"),
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(self.positions)

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
            pos = self.robot_node.getField("translation").getSFVec3f()
            self.positions.append(pos[:2])

            action = self.get_action()
            if action.shape != (2,):
                self.save_positions(it)
                break

            self.update_motors(action)

            """Observation sent to server is lidar readings + collision/end flag"""
            lidar = self.read_observation()
            collided = self.detect_collision()
            end = self.detect_end()

            if collided or end:
                self.save_positions(it)
                self.positions = []
                it += 1
                self.reset_robot()

            state = RobotState(
                lidar=lidar,
                prev_action=0,  # placeholder, will be set in env
                collided=collided,
                goal_reached=end,
            )

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

    def update_motors(self, action: np.ndarray) -> None:
        """
        Action is pair linear velocity, angular velocity
        Convert to left and right wheel speeds
        """
        left_speed = action[0] - action[1] * self.l / 2
        right_speed = action[0] + action[1] * self.l / 2
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

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
