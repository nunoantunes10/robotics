import os
import argparse
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor
import csv

TIME_STEPS = 5_000
N_ROBOTS = 9
LIDAR_DIM = 360


def patch_numpy_pickle_modules():
    """Allow NumPy 2.x pickles to load in NumPy 1.x environments."""
    if not hasattr(np, "_core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.numeric"] = np.core.numeric
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        sys.modules["numpy._core.umath"] = np.core.umath


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained PPO model in Webots.")
    parser.add_argument(
        "-n",
        "--robots",
        type=int,
        default=N_ROBOTS,
        help=f"Number of robot clients to test. Default: {N_ROBOTS}",
    )
    return parser.parse_args()


def run_model(n_robots=N_ROBOTS):
    """Start vectorized environment to test model in parallel"""

    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i, lidar_dim=LIDAR_DIM))
        return _init

    env = DummyVecEnv([env_fn(i) for i in range(n_robots)])
    path = f"./models/ppo_wheelchair_lidar{LIDAR_DIM}_human"
    vecnorm_path = f"./models/vecnormalize_lidar{LIDAR_DIM}_human.pkl"
    patch_numpy_pickle_modules()
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    assert os.path.exists(
        path + ".zip"
    ), "Model path does not exist. Please train the model first."

    model = PPO.load(path, env)

    success_counts = [0] * n_robots
    episode_counts = [0] * n_robots
    episode_rewards = [0.0] * n_robots
    episode_rows = []

    obs = env.reset()
    for _ in range(TIME_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        for i in range(n_robots):
            episode_rewards[i] += float(rewards[i])

        for i, done in enumerate(dones):
            if done:
                episode_counts[i] += 1
                success = infos[i].get("is_success", False)
                collision = infos[i].get("collision", False)
                goal_reached = infos[i].get("goal_reached", False)
                time_step = infos[i].get("time_step", 0)
                final_step_reward = infos[i].get("reward", 0.0)

                if success:
                    success_counts[i] += 1

                episode_rows.append([
                    i,
                    episode_counts[i],
                    int(success),
                    int(collision),
                    int(goal_reached),
                    time_step,
                    episode_rewards[i],
                    final_step_reward,
                    LIDAR_DIM,
                ])

                episode_rewards[i] = 0.0

    for i in range(n_robots):
        if episode_counts[i] > 0:
            rate = 100 * success_counts[i] / episode_counts[i]
            print(
                f"Robot {i}: {success_counts[i]}/{episode_counts[i]} successes ({rate:.1f}%)"
            )
        else:
            print(f"Robot {i}: no completed episodes.")

    with open("success_rates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["robot_id", "successes", "episodes", "success_rate", "lidar_dim"])
        for i in range(n_robots):
            rate = (
                100 * success_counts[i] / episode_counts[i] if episode_counts[i] else 0
            )
            writer.writerow([i, success_counts[i], episode_counts[i], f"{rate:.2f}", LIDAR_DIM])

    with open("episode_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "robot_id",
            "episode_id",
            "success",
            "collision",
            "goal_reached",
            "episode_steps",
            "episode_reward",
            "final_step_reward",
            "lidar_dim",
        ])
        writer.writerows(episode_rows)


if __name__ == "__main__":
    args = parse_args()
    run_model(n_robots=args.robots)
