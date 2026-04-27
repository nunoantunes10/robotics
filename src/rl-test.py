import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor
import csv

TIME_STEPS = 5_000
N_ROBOTS = 9


def run_model():
    """Start vectorized environment to train model in parallel"""

    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i))

        return _init

    env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    path = "./models/ppo_wheelchair"
    assert os.path.exists(
        path + ".zip"
    ), "Model path does not exist. Please train the model first."

    model = PPO.load(path, env)

    """
    Test the model
    """
    success_counts = [0] * N_ROBOTS
    episode_counts = [0] * N_ROBOTS

    obs = env.reset()
    for _ in range(TIME_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        for i, done in enumerate(dones):
            if done:
                episode_counts[i] += 1
                if infos[i].get("is_success", False):
                    success_counts[i] += 1

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
        writer.writerow(["robot_id", "successes", "episodes", "success_rate"])
        for i in range(N_ROBOTS):
            rate = (
                100 * success_counts[i] / episode_counts[i] if episode_counts[i] else 0
            )
            writer.writerow([i, success_counts[i], episode_counts[i], f"{rate:.2f}"])


if __name__ == "__main__":
    run_model()
