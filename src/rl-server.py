import argparse
import os
import traceback
from stable_baselines3.common.vec_env import DummyVecEnv
from wheelchair_env import WheelchairEnv
from stable_baselines3 import PPO
from cnn_feature_extractor import LidarCNNFeatureExtractor
from lstm_feature_extractor import LidarLSTMFeatureExtractor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from training_metrics import TrainingMetricsCallback
from torch import cuda

TRAIN_STEPS = 10000000
N_ROBOTS = 9
LIDAR_DIM = 360


def get_training_device():
    device = "cuda" if cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Training device: CUDA ({cuda.get_device_name(0)})")
    else:
        print("Training device: CPU (CUDA not available)")
    return device


def get_model_config(nn_type: str):
    configs = {
        "cnn": {
            "extractor": LidarCNNFeatureExtractor,
            "path": f"./models/ppo_wheelchair_cnn_goal_v2_lidar{LIDAR_DIM}",
            "best_path": f"./models/ppo_wheelchair_cnn_goal_v2_lidar{LIDAR_DIM}_best",
            "vecnorm_path": f"./models/vecnormalize_cnn_goal_v2_lidar{LIDAR_DIM}.pkl",
            "best_vecnorm_path": f"./models/vecnormalize_cnn_goal_v2_lidar{LIDAR_DIM}_best.pkl",
            "tb_log_name": f"ppo-cnn-goal-v2-lidar{LIDAR_DIM}",
            "legacy_path": None,
            "legacy_vecnorm_path": None,
            "ppo_kwargs": {
                "n_steps": 512,
                "learning_rate": 3e-5,
                "batch_size": 1024,
                "n_epochs": 8,
                "clip_range": 0.08,
                "ent_coef": 0.003,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "target_kl": 0.02,
            },
        },
        "lstm": {
            "extractor": LidarLSTMFeatureExtractor,
            "path": f"./models/ppo_wheelchair_lstm_goal_v3_lidar{LIDAR_DIM}",
            "best_path": f"./models/ppo_wheelchair_lstm_goal_v3_lidar{LIDAR_DIM}_best",
            "vecnorm_path": f"./models/vecnormalize_lstm_goal_v3_lidar{LIDAR_DIM}.pkl",
            "best_vecnorm_path": f"./models/vecnormalize_lstm_goal_v3_lidar{LIDAR_DIM}_best.pkl",
            "tb_log_name": f"ppo-lstm-goal-v3-lidar{LIDAR_DIM}",
            "legacy_path": None,
            "legacy_vecnorm_path": None,
            "ppo_kwargs": {
                "n_steps": 256,
                "learning_rate": 2.5e-5,
                "batch_size": 512,
                "n_epochs": 6,
                "clip_range": 0.05,
                "ent_coef": 0.001,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "target_kl": 0.02,
            },
        },
    }
    return configs[nn_type]


def existing_path(path: str, legacy_path: str | None = None) -> str | None:
    if os.path.exists(path + ".zip"):
        return path
    if legacy_path is not None and os.path.exists(legacy_path + ".zip"):
        return legacy_path
    return None


def existing_vecnorm_path(path: str, legacy_path: str | None = None) -> str | None:
    if os.path.exists(path):
        return path
    if legacy_path is not None and os.path.exists(legacy_path):
        return legacy_path
    return None


def train_model(new=False, nn_type="cnn"):
    env = None
    model = None
    metrics_callback = None

    try:
        """Start vectorized environment to train model in parallel"""
        config = get_model_config(nn_type)
        device = get_training_device()

        def env_fn(i):
            def _init():
                return Monitor(WheelchairEnv(i, lidar_dim=LIDAR_DIM))

            return _init

        path = config["path"]
        vecnorm_path = config["vecnorm_path"]
        model_load_path = existing_path(path, config["legacy_path"])
        vecnorm_load_path = existing_vecnorm_path(
            vecnorm_path,
            config["legacy_vecnorm_path"],
        )
        prev_model = model_load_path is not None

        env = DummyVecEnv([env_fn(i) for i in range(N_ROBOTS)])
        if vecnorm_load_path is not None and not new:
            print(f"Loading VecNormalize stats from {vecnorm_load_path}")
            env = VecNormalize.load(vecnorm_load_path, env)
            env.training = True
            env.norm_obs = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=False)

        if prev_model and not new:
            print(f"Loading previous {nn_type.upper()} model from {model_load_path}")
            model = PPO.load(model_load_path, env=env, device=device)
        else:
            print(f"Creating new {nn_type.upper()} model")

            policy_kwargs = dict(
                features_extractor_class=config["extractor"],
                features_extractor_kwargs=dict(features_dim=128),
            )
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device,
                tensorboard_log="logs",
                **config["ppo_kwargs"],
            )

        print(f"Training target timesteps: {TRAIN_STEPS}")
        print(f"Saving checkpoints to {path}.zip")
        print(f"Saving VecNormalize stats to {vecnorm_path}")
        print(f"Saving best checkpoint to {config['best_path']}.zip")
        metrics_callback = TrainingMetricsCallback(
            nn_type=nn_type,
            lidar_dim=LIDAR_DIM,
            best_model_path=config["best_path"],
            best_vecnorm_path=config["best_vecnorm_path"],
        )
        model.learn(
            total_timesteps=TRAIN_STEPS,
            tb_log_name=config["tb_log_name"],
            callback=metrics_callback,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception:
        print("Training stopped because of an exception:")
        traceback.print_exc()
        raise
    finally:
        if metrics_callback is not None:
            print("Saving training metrics and plots")
            metrics_callback.save()

        if model is not None and env is not None:
            print("Saving model and VecNormalize stats")
            model.save(path)
            env.save(vecnorm_path)

        if env is not None:
            print("Calling env.close()")
            env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a fresh model for the selected network instead of loading a checkpoint.",
    )
    parser.add_argument(
        "--nn",
        choices=["cnn", "lstm"],
        default="cnn",
        help="Feature extractor network to use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(new=args.new, nn_type=args.nn)
