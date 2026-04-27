import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor
import csv

TIME_STEPS = 5_000
N_ROBOTS = 9
LIDAR_DIM = 360


def run_model():
    """Start vectorized environment to test model in parallel"""

    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i, lidar_dim=LIDAR_DIM))
        return _init

    env = DummyVecEnv([env_fn(i) for i in range(N_ROBOTS)])
    path = f"./models/ppo_wheelchair_lidar{LIDAR_DIM}"
    vecnorm_path = f"./models/vecnormalize_lidar{LIDAR_DIM}.pkl"
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    assert os.path.exists(
        path + ".zip"
    ), "Model path does not exist. Please train the model first."

    model = PPO.load(path, env)

    success_counts = [0] * N_ROBOTS
    episode_counts = [0] * N_ROBOTS
    episode_rewards = [0.0] * N_ROBOTS
    episode_rows = []

    obs = env.reset()
    for _ in range(TIME_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        for i in range(N_ROBOTS):
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

    for i in range(N_ROBOTS):
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
        for i in range(N_ROBOTS):
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
    run_model()
