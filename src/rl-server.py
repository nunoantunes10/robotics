import os
import sys
from stable_baselines3.common.vec_env import DummyVecEnv
from wheelchair_env import WheelchairEnv
from stable_baselines3 import PPO
from cnn_feature_extractor import LidarCNNFeatureExtractor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor

TRAIN_STEPS = 20000
N_ROBOTS = 9
LIDAR_DIM = 90


def train_model(new=False):
    env = None

    try:
        """Start vectorized environment to train model in parallel"""

        def env_fn(i):
            def _init():
                return Monitor(WheelchairEnv(i, lidar_dim=LIDAR_DIM))

            return _init

        env = DummyVecEnv([env_fn(i) for i in range(N_ROBOTS)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

        path = f"./models/ppo_wheelchair_lidar{LIDAR_DIM}"
        prev_model = os.path.exists(path + ".zip")

        if prev_model and not new:
            print("Loading previous model")
            model = PPO.load(path, env=env)
        else:
            if prev_model:
                print("Deleting previous model")
                os.remove(path + ".zip")

            print("Creating new model")

            policy_kwargs = dict(
                features_extractor_class=LidarCNNFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                n_steps=128,
                learning_rate=5e-5,
                batch_size=1024,
                n_epochs=20,
                clip_range=0.1,
                ent_coef=0.01,
                device="cpu",
                tensorboard_log="logs",
            )

        model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=f"ppo-lidar{LIDAR_DIM}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Calling env.close()")
        env.close()
        model.save(path)


if __name__ == "__main__":
    new = sys.argv[1] == "--new" if len(sys.argv) > 1 else False
    train_model(new)
