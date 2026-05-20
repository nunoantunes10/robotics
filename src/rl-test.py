import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor
import csv

TIME_STEPS = 5_000
N_ROBOTS = 9
LIDAR_DIM = 360


def get_model_paths(nn_type: str):
    paths = {
        "cnn": {
            "path": f"./models/ppo_wheelchair_cnn_goal_v2_lidar{LIDAR_DIM}",
            "best_path": f"./models/ppo_wheelchair_cnn_goal_v2_lidar{LIDAR_DIM}_best",
            "vecnorm_path": f"./models/vecnormalize_cnn_goal_v2_lidar{LIDAR_DIM}.pkl",
            "best_vecnorm_path": f"./models/vecnormalize_cnn_goal_v2_lidar{LIDAR_DIM}_best.pkl",
        },
        "lstm": {
            "path": f"./models/ppo_wheelchair_lstm_goal_v3_lidar{LIDAR_DIM}",
            "best_path": f"./models/ppo_wheelchair_lstm_goal_v3_lidar{LIDAR_DIM}_best",
            "vecnorm_path": f"./models/vecnormalize_lstm_goal_v3_lidar{LIDAR_DIM}.pkl",
            "best_vecnorm_path": f"./models/vecnormalize_lstm_goal_v3_lidar{LIDAR_DIM}_best.pkl",
        },
    }
    config = paths[nn_type]
    if (
        os.path.exists(config["best_path"] + ".zip")
        and os.path.exists(config["best_vecnorm_path"])
    ):
        return config["best_path"], config["best_vecnorm_path"], True
    return config["path"], config["vecnorm_path"], False


def run_model(nn_type="cnn"):
    """Start vectorized environment to test model in parallel"""

    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i, lidar_dim=LIDAR_DIM))
        return _init

    env = DummyVecEnv([env_fn(i) for i in range(N_ROBOTS)])
    path, vecnorm_path, best_checkpoint = get_model_paths(nn_type)

    if not os.path.exists(path + ".zip"):
        raise FileNotFoundError(f"Model path does not exist: {path}.zip")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize path does not exist: {vecnorm_path}")

    checkpoint_label = "best" if best_checkpoint else "final"
    print(f"Testing {checkpoint_label} {nn_type.upper()} model from {path}.zip")
    print(f"Loading VecNormalize stats from {vecnorm_path}")

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_obs = False
    env.norm_reward = False

    model = PPO.load(path, env=env)

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

    success_path = f"success_rates_{nn_type}.csv"
    episode_path = f"episode_metrics_{nn_type}.csv"

    with open(success_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["robot_id", "successes", "episodes", "success_rate", "lidar_dim"])
        for i in range(N_ROBOTS):
            rate = (
                100 * success_counts[i] / episode_counts[i] if episode_counts[i] else 0
            )
            writer.writerow([i, success_counts[i], episode_counts[i], f"{rate:.2f}", LIDAR_DIM])

    with open(episode_path, "w", newline="") as f:
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
    print(f"Wrote {success_path}")
    print(f"Wrote {episode_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nn",
        choices=["cnn", "lstm"],
        default="cnn",
        help="Model type to test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_model(nn_type=args.nn)
