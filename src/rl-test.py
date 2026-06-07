import os
import argparse
import sys
import re
import csv
import glob
import json
import shutil
import time
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor

TIME_STEPS = 5_000
N_ROBOTS = 9
LIDAR_DIM = 360
PROGRESS_LOG_INTERVAL = 1_000
LOG_DIR = "logs"


def safe_label(value):
    label = os.path.splitext(os.path.basename(value))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "unknown"


def archive_flat_trajectory_logs():
    paths = sorted(glob.glob(os.path.join(LOG_DIR, "positions_*.csv")))
    if not paths:
        return None

    archive_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = os.path.join(LOG_DIR, "legacy_flat_logs", archive_id)
    os.makedirs(archive_dir, exist_ok=True)
    for path in paths:
        shutil.move(path, os.path.join(archive_dir, os.path.basename(path)))
    return archive_dir


def create_run_paths(world, model_path):
    model_name = safe_label(model_path)
    world_name = safe_label(world)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(LOG_DIR, "runs", f"{run_id}__{world_name}__{model_name}")
    trajectory_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "trajectory_dir": trajectory_dir,
        "world_name": world_name,
        "model_name": model_name,
        "success_rates": os.path.join(run_dir, "success_rates.csv"),
        "episode_metrics": os.path.join(run_dir, "episode_metrics.csv"),
        "run_summary": os.path.join(run_dir, "run_summary.csv"),
        "metadata": os.path.join(run_dir, "metadata.json"),
    }


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
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=TIME_STEPS,
        help=f"Number of environment steps to run. Default: {TIME_STEPS}",
    )
    parser.add_argument(
        "-w",
        "--world",
        default="unknown_world",
        help="World name or .wbt path used for this test run.",
    )
    return parser.parse_args()


def save_outputs(
    run_paths,
    metadata,
    success_counts,
    episode_counts,
    target_times,
    episode_rows,
):
    all_target_times = [
        time_step for robot_times in target_times for time_step in robot_times
    ]
    total_successes = sum(success_counts)
    total_episodes = sum(episode_counts)
    overall_success_rate = (
        100 * total_successes / total_episodes if total_episodes else 0
    )
    avg_time_to_target = (
        sum(all_target_times) / len(all_target_times) if all_target_times else 0.0
    )
    min_time_to_target = min(all_target_times) if all_target_times else 0.0
    max_time_to_target = max(all_target_times) if all_target_times else 0.0

    with open(run_paths["success_rates"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "world",
            "model",
            "robot_id",
            "successes",
            "episodes",
            "success_rate",
            "avg_time_to_target",
            "lidar_dim",
        ])
        for i in range(len(success_counts)):
            rate = (
                100 * success_counts[i] / episode_counts[i] if episode_counts[i] else 0
            )
            robot_avg_time = (
                sum(target_times[i]) / len(target_times[i]) if target_times[i] else 0.0
            )
            writer.writerow([
                run_paths["run_id"],
                run_paths["world_name"],
                run_paths["model_name"],
                i,
                success_counts[i],
                episode_counts[i],
                f"{rate:.2f}",
                f"{robot_avg_time:.2f}",
                LIDAR_DIM,
            ])

    with open(run_paths["episode_metrics"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "world",
            "model",
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

    with open(run_paths["run_summary"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "world",
            "model",
            "successful_episodes",
            "total_episodes",
            "success_rate",
            "avg_time_to_target",
            "min_time_to_target",
            "max_time_to_target",
            "lidar_dim",
            "completed_timesteps",
            "requested_timesteps",
            "interrupted",
        ])
        writer.writerow([
            run_paths["run_id"],
            run_paths["world_name"],
            run_paths["model_name"],
            total_successes,
            total_episodes,
            f"{overall_success_rate:.2f}",
            f"{avg_time_to_target:.2f}",
            f"{min_time_to_target:.2f}",
            f"{max_time_to_target:.2f}",
            LIDAR_DIM,
            metadata["completed_timesteps"],
            metadata["requested_timesteps"],
            int(metadata["interrupted"]),
        ])

    metadata["output_files"] = {
        "success_rates": run_paths["success_rates"],
        "episode_metrics": run_paths["episode_metrics"],
        "run_summary": run_paths["run_summary"],
        "metadata": run_paths["metadata"],
        "trajectories": run_paths["trajectory_dir"],
    }
    with open(run_paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)


def move_flat_trajectory_logs(trajectory_dir):
    os.makedirs(trajectory_dir, exist_ok=True)
    time.sleep(0.5)
    moved = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "positions_*.csv"))):
        destination = os.path.join(trajectory_dir, os.path.basename(path))
        shutil.move(path, destination)
        moved.append(destination)
    return moved


def run_model(n_robots=N_ROBOTS, time_steps=TIME_STEPS, world="unknown_world"):
    """Start vectorized environment to test model in parallel"""

    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i, lidar_dim=LIDAR_DIM))
        return _init

    path = f"./models/ppo_wheelchair_lidar{LIDAR_DIM}_human"
    vecnorm_path = f"./models/vecnormalize_lidar{LIDAR_DIM}_human.pkl"
    os.makedirs(LOG_DIR, exist_ok=True)
    archive_dir = archive_flat_trajectory_logs()
    run_paths = create_run_paths(world, path)
    metadata = {
        "run_id": run_paths["run_id"],
        "timestamp": run_paths["run_id"],
        "world": run_paths["world_name"],
        "world_arg": world,
        "model": run_paths["model_name"],
        "model_path": path + ".zip",
        "vecnormalize_path": vecnorm_path,
        "robots": n_robots,
        "requested_timesteps": time_steps,
        "completed_timesteps": 0,
        "interrupted": False,
        "lidar_dim": LIDAR_DIM,
        "archived_legacy_logs": archive_dir,
        "moved_trajectories": [],
        "output_files": {},
    }

    env = None

    print(f"Saving this test run to {run_paths['run_dir']}")
    if archive_dir:
        print(f"Archived existing flat trajectory logs to {archive_dir}")

    success_counts = [0] * n_robots
    episode_counts = [0] * n_robots
    episode_rewards = [0.0] * n_robots
    target_times = [[] for _ in range(n_robots)]
    episode_rows = []

    try:
        env = DummyVecEnv([env_fn(i) for i in range(n_robots)])
        patch_numpy_pickle_modules()
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

        assert os.path.exists(
            path + ".zip"
        ), "Model path does not exist. Please train the model first."

        model = PPO.load(path, env)

        obs = env.reset()
        for current_step in range(1, time_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            metadata["completed_timesteps"] = current_step

            if current_step % PROGRESS_LOG_INTERVAL == 0 or current_step == time_steps:
                progress = 100 * current_step / time_steps
                print(
                    f"Progress: {current_step}/{time_steps} timesteps ({progress:.1f}%)"
                )

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
                        target_times[i].append(time_step)

                    episode_rows.append([
                        run_paths["run_id"],
                        run_paths["world_name"],
                        run_paths["model_name"],
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
    except KeyboardInterrupt:
        metadata["interrupted"] = True
        print("\nTest interrupted. Saving results collected so far...")
    finally:
        if env is not None:
            env.close()

        metadata["moved_trajectories"] = move_flat_trajectory_logs(
            run_paths["trajectory_dir"]
        )

        for i in range(n_robots):
            avg_time_to_target = (
                sum(target_times[i]) / len(target_times[i]) if target_times[i] else None
            )
            if episode_counts[i] > 0:
                rate = 100 * success_counts[i] / episode_counts[i]
                avg_time_text = (
                    f"{avg_time_to_target:.2f} steps"
                    if avg_time_to_target is not None
                    else "n/a"
                )
                print(
                    f"Robot {i}: {success_counts[i]}/{episode_counts[i]} successes "
                    f"({rate:.1f}%), avg time to target: {avg_time_text}"
                )
            else:
                print(f"Robot {i}: no completed episodes, avg time to target: n/a")

        save_outputs(
            run_paths,
            metadata,
            success_counts,
            episode_counts,
            target_times,
            episode_rows,
        )

        print(f"Saved success rates to {run_paths['success_rates']}")
        print(f"Saved episode metrics to {run_paths['episode_metrics']}")
        print(f"Saved run summary to {run_paths['run_summary']}")
        print(f"Saved metadata to {run_paths['metadata']}")


if __name__ == "__main__":
    args = parse_args()
    run_model(n_robots=args.robots, time_steps=args.timesteps, world=args.world)
