import argparse
import json
import os
import re
import sys
from datetime import datetime


TRAIN_STEPS = 3_000_000
N_ROBOTS = 9
LIDAR_DIM = 360
DEFAULT_REWARD_MODE = "transfer_finetune"
STRAIGHT_BONUS_CLEAR = 0.25
SOFT_FORWARD_BONUS_CLEAR = 0.08
TURN_PENALTY_CLEAR = 0.15
STOP_PENALTY_CLEAR = 0.10
OSCILLATION_PENALTY = 0.20
STRONG_OSCILLATION_PENALTY = 0.40
NON_FORWARD_CLEAR_PENALTY = 0.30
CLEAR_FRONT_THRESHOLD = 1.5
DANGER_FRONT_THRESHOLD = 0.75
GOAL_DIRECTION_PROGRESS_WEIGHT = 2.0
GOAL_DIRECTION_PROGRESS_CLIP = 0.5
GOAL_DIRECTION_BACKWARD_PENALTY = 0.10
GOAL_DIRECTION_LOOP_PENALTY = 0.05
GOAL_DIRECTION_LOOP_WINDOW = 20
GOAL_DIRECTION_MIN_RECENT_PROGRESS = 0.02
GOAL_DIRECTION_CLEAR_FRONT_THRESHOLD = 1.5
GOAL_DIRECTION_HEADING_WEIGHT = 0.05
DEFAULT_WORLD = "worlds/smart-wheelchairs-transfer-finetune.wbt"
DEFAULT_SOURCE_MODEL = f"models/ppo_wheelchair_lidar{LIDAR_DIM}_human"
DEFAULT_SOURCE_VECNORMALIZE = f"models/vecnormalize_lidar{LIDAR_DIM}_human.pkl"
DEFAULT_OUTPUT_MODEL = f"models/ppo_wheelchair_lidar{LIDAR_DIM}_human_transfer_finetune"
DEFAULT_OUTPUT_VECNORMALIZE = (
    f"models/vecnormalize_lidar{LIDAR_DIM}_human_transfer_finetune.pkl"
)
DEFAULT_OUTPUT_DIR = "outputs/experiments"
DEFAULT_TENSORBOARD_LOG = None


def with_zip_suffix(path):
    return path if path.endswith(".zip") else path + ".zip"


def safe_label(value):
    label = os.path.splitext(os.path.basename(value))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "experiment"


def make_experiment_dir(output_dir, experiment_name):
    if experiment_name:
        experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = os.path.join(
            output_dir, f"{experiment_id}__{safe_label(experiment_name)}"
        )
    else:
        experiment_dir = output_dir

    os.makedirs(os.path.join(experiment_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "csv"), exist_ok=True)
    return experiment_dir


def write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def transfer_config(args, robot_ids, experiment_dir, phase, start_time, end_time=None):
    source_model_zip = with_zip_suffix(args.source_model)
    output_model_zip = with_zip_suffix(args.output_model)
    return {
        "phase": phase,
        "start_time": start_time,
        "end_time": end_time,
        "experiment_name": args.experiment_name,
        "experiment_dir": experiment_dir,
        "world": args.world,
        "robot_ids": robot_ids,
        "robots": args.robots,
        "lidar_dim": args.lidar_dim,
        "source_model": source_model_zip,
        "source_vecnormalize": args.source_vecnormalize,
        "output_model": output_model_zip,
        "output_vecnormalize": args.output_vecnormalize,
        "tensorboard_log": args.resolved_tensorboard_log,
        "tb_log_name": args.resolved_tb_log_name,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "reward_mode": args.reward_mode,
        "reward_coefficients": {
            "straight_bonus": args.straight_bonus,
            "soft_forward_bonus": args.soft_forward_bonus,
            "turn_penalty_clear": args.turn_penalty_clear,
            "stop_penalty_clear": args.stop_penalty_clear,
            "oscillation_penalty": args.oscillation_penalty,
            "strong_oscillation_penalty": args.strong_oscillation_penalty,
            "non_forward_clear_penalty": args.non_forward_clear_penalty,
            "clear_front_threshold": args.clear_front_threshold,
            "danger_front_threshold": args.danger_front_threshold,
            "goal_direction_progress_weight": args.goal_direction_progress_weight,
            "goal_direction_progress_clip": args.goal_direction_progress_clip,
            "goal_direction_backward_penalty": args.goal_direction_backward_penalty,
            "goal_direction_loop_penalty": args.goal_direction_loop_penalty,
            "goal_direction_loop_window": args.goal_direction_loop_window,
            "goal_direction_min_recent_progress": (
                args.goal_direction_min_recent_progress
            ),
            "goal_direction_clear_front_threshold": (
                args.goal_direction_clear_front_threshold
            ),
            "goal_direction_heading_weight": args.goal_direction_heading_weight,
        },
        "overwrite_output": args.overwrite_output,
        "check_only": args.check_only,
    }


def parse_robot_client_ids(world_path):
    with open(world_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    robot_ids = []
    inside_robot_client = False
    inside_args = False

    for line in lines:
        stripped = line.strip()
        if stripped == 'controller "robot_client"':
            inside_robot_client = True
            continue

        if inside_robot_client and stripped.startswith("controllerArgs"):
            inside_args = True
            continue

        if inside_robot_client and inside_args:
            match = re.fullmatch(r'"?(\d+)"?', stripped)
            if match:
                robot_ids.append(int(match.group(1)))
            if stripped == "]":
                inside_robot_client = False
                inside_args = False

    return sorted(robot_ids)


def validate_paths(args):
    source_model_zip = with_zip_suffix(args.source_model)
    output_model_zip = with_zip_suffix(args.output_model)

    if not os.path.exists(args.world):
        raise FileNotFoundError(f"World file does not exist: {args.world}")

    if not os.path.exists(source_model_zip):
        raise FileNotFoundError(f"Source model does not exist: {source_model_zip}")

    if os.path.abspath(source_model_zip) == os.path.abspath(output_model_zip):
        raise ValueError(
            "Refusing to fine-tune in place: output model path matches source model."
        )

    if os.path.abspath(args.source_vecnormalize) == os.path.abspath(
        args.output_vecnormalize
    ):
        raise ValueError(
            "Refusing to fine-tune in place: output VecNormalize path matches source "
            "VecNormalize path."
        )

    existing_outputs = [
        path
        for path in [output_model_zip, args.output_vecnormalize]
        if os.path.exists(path)
    ]
    if existing_outputs and not args.overwrite_output and not args.check_only:
        existing_text = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "Output files already exist. Pass --overwrite-output to replace them:\n"
            f"{existing_text}"
        )


def validate_world_robot_count(world_path, requested_robots):
    robot_ids = parse_robot_client_ids(world_path)
    expected_ids = list(range(requested_robots))

    if not robot_ids:
        raise ValueError(f"No robot_client controllerArgs found in {world_path}")

    missing_ids = [robot_id for robot_id in expected_ids if robot_id not in robot_ids]
    if missing_ids:
        raise ValueError(
            f"World {world_path} has robot client IDs {robot_ids}, but requested "
            f"{requested_robots} robots requires IDs {expected_ids}. Missing: "
            f"{missing_ids}"
        )

    return robot_ids


def patch_numpy_pickle_modules():
    """Allow NumPy 2.x pickles to load in NumPy 1.x environments."""
    import numpy as np

    if not hasattr(np, "_core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.numeric"] = np.core.numeric
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        sys.modules["numpy._core.umath"] = np.core.umath


def describe_space(space):
    description = f"{space.__class__.__name__}"
    if hasattr(space, "shape"):
        description += f"(shape={space.shape}, dtype={getattr(space, 'dtype', None)})"
    if hasattr(space, "n"):
        description += f"(n={space.n}, start={getattr(space, 'start', 0)})"
    return description


def spaces_match(left, right):
    import numpy as np

    if left.__class__ is not right.__class__:
        return False

    if getattr(left, "shape", None) != getattr(right, "shape", None):
        return False

    if hasattr(left, "dtype") and getattr(left, "dtype", None) != getattr(
        right, "dtype", None
    ):
        return False

    if hasattr(left, "low") and not np.array_equal(left.low, right.low):
        return False

    if hasattr(left, "high") and not np.array_equal(left.high, right.high):
        return False

    if hasattr(left, "n") and left.n != right.n:
        return False

    if hasattr(left, "start") and left.start != right.start:
        return False

    return True


def assert_model_env_spaces_match(model, env):
    model_obs = model.observation_space
    model_action = model.action_space
    env_obs = env.observation_space
    env_action = env.action_space

    mismatches = []
    if not spaces_match(model_obs, env_obs):
        mismatches.append(
            "observation space mismatch: "
            f"model={describe_space(model_obs)}, env={describe_space(env_obs)}"
        )

    if not spaces_match(model_action, env_action):
        mismatches.append(
            "action space mismatch: "
            f"model={describe_space(model_action)}, env={describe_space(env_action)}"
        )

    if mismatches:
        raise ValueError(
            "Loaded model is not compatible with the new environment:\n"
            + "\n".join(f"  - {mismatch}" for mismatch in mismatches)
        )


def set_model_learning_rate(model, learning_rate):
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _: learning_rate
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = learning_rate


def make_env(env_id, args):
    from stable_baselines3.common.monitor import Monitor
    from wheelchair_env import WheelchairEnv

    def _init():
        return Monitor(
            WheelchairEnv(
                env_id,
                lidar_dim=args.lidar_dim,
                reward_mode=args.reward_mode,
                straight_bonus_clear=args.straight_bonus,
                soft_forward_bonus_clear=args.soft_forward_bonus,
                turn_penalty_clear=args.turn_penalty_clear,
                stop_penalty_clear=args.stop_penalty_clear,
                oscillation_penalty=args.oscillation_penalty,
                strong_oscillation_penalty=args.strong_oscillation_penalty,
                non_forward_clear_penalty=args.non_forward_clear_penalty,
                clear_front_threshold=args.clear_front_threshold,
                danger_front_threshold=args.danger_front_threshold,
                goal_direction_progress_weight=args.goal_direction_progress_weight,
                goal_direction_progress_clip=args.goal_direction_progress_clip,
                goal_direction_backward_penalty=args.goal_direction_backward_penalty,
                goal_direction_loop_penalty=args.goal_direction_loop_penalty,
                goal_direction_loop_window=args.goal_direction_loop_window,
                goal_direction_min_recent_progress=(
                    args.goal_direction_min_recent_progress
                ),
                goal_direction_clear_front_threshold=(
                    args.goal_direction_clear_front_threshold
                ),
                goal_direction_heading_weight=args.goal_direction_heading_weight,
            )
        )

    return _init


def fine_tune(args):
    validate_paths(args)
    robot_ids = validate_world_robot_count(args.world, args.robots)
    experiment_dir = make_experiment_dir(args.output_dir, args.experiment_name)
    start_time = datetime.now().isoformat(timespec="seconds")
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.resolved_tensorboard_log = args.tensorboard_log or os.path.join(
        experiment_dir, "training", "tensorboard"
    )
    args.resolved_tb_log_name = (
        f"{run_stamp}__ppo-lidar{args.lidar_dim}-{safe_label(args.reward_mode)}"
    )

    source_model_zip = with_zip_suffix(args.source_model)
    output_model_zip = with_zip_suffix(args.output_model)

    print("Transfer fine-tuning configuration")
    print(f"  World/map: {args.world}")
    print(f"  Requested robots: {args.robots}")
    print(f"  Robot client IDs found in world: {robot_ids}")
    print(f"  Source model: {source_model_zip}")
    print(f"  Source VecNormalize: {args.source_vecnormalize}")
    print(f"  Output model: {output_model_zip}")
    print(f"  Output VecNormalize: {args.output_vecnormalize}")
    print(f"  Experiment dir: {experiment_dir}")
    print(f"  TensorBoard log dir: {args.resolved_tensorboard_log}")
    print(f"  TensorBoard run name: {args.resolved_tb_log_name}")
    print(f"  LiDAR dim: {args.lidar_dim}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Reward mode: {args.reward_mode}")
    print("  Transfer reward coefficients:")
    print(f"    straight_bonus: {args.straight_bonus}")
    print(f"    soft_forward_bonus: {args.soft_forward_bonus}")
    print(f"    turn_penalty_clear: {args.turn_penalty_clear}")
    print(f"    stop_penalty_clear: {args.stop_penalty_clear}")
    print(f"    oscillation_penalty: {args.oscillation_penalty}")
    print(f"    strong_oscillation_penalty: {args.strong_oscillation_penalty}")
    print(f"    non_forward_clear_penalty: {args.non_forward_clear_penalty}")
    print(f"    clear_front_threshold: {args.clear_front_threshold}")
    print(f"    danger_front_threshold: {args.danger_front_threshold}")
    print("  Goal-direction reward coefficients:")
    print(f"    progress_weight: {args.goal_direction_progress_weight}")
    print(f"    progress_clip: {args.goal_direction_progress_clip}")
    print(f"    backward_penalty: {args.goal_direction_backward_penalty}")
    print(f"    loop_penalty: {args.goal_direction_loop_penalty}")
    print(f"    loop_window: {args.goal_direction_loop_window}")
    print(f"    min_recent_progress: {args.goal_direction_min_recent_progress}")
    print(f"    clear_front_threshold: {args.goal_direction_clear_front_threshold}")
    print(f"    heading_weight: {args.goal_direction_heading_weight}")
    print("  Original model will not be overwritten.")

    config_path = os.path.join(experiment_dir, "training", "transfer_config.json")
    experiment_path = os.path.join(experiment_dir, "metadata", "experiment.json")
    write_json(
        config_path,
        transfer_config(
            args,
            robot_ids,
            experiment_dir,
            "check_only" if args.check_only else "started",
            start_time,
        ),
    )
    write_json(
        experiment_path,
        {
            "experiment_name": args.experiment_name,
            "experiment_dir": experiment_dir,
            "created_at": start_time,
            "world": args.world,
            "source_model": source_model_zip,
            "finetuned_model": output_model_zip,
            "source_vecnormalize": args.source_vecnormalize,
            "finetuned_vecnormalize": args.output_vecnormalize,
            "report_path": os.path.join(experiment_dir, "report.html"),
            "paths": {
                "metadata": os.path.join(experiment_dir, "metadata"),
                "training": os.path.join(experiment_dir, "training"),
                "evaluation": os.path.join(experiment_dir, "evaluation"),
                "plots": os.path.join(experiment_dir, "plots"),
                "csv": os.path.join(experiment_dir, "csv"),
            },
        },
    )

    if args.check_only:
        print(f"Wrote experiment metadata to {experiment_path}")
        print(f"Wrote transfer config to {config_path}")
        print("Check-only mode complete; no environment was created and no training ran.")
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = None
    model = None
    training_started = False
    training_completed = False
    interrupted = False
    try:
        env = DummyVecEnv([make_env(i, args) for i in range(args.robots)])
        patch_numpy_pickle_modules()

        if os.path.exists(args.source_vecnormalize):
            print(f"Loading VecNormalize stats from {args.source_vecnormalize}")
            env = VecNormalize.load(args.source_vecnormalize, env)
            env.training = True
            env.norm_reward = False
            print("VecNormalize loaded and set to training mode.")
        else:
            print(
                f"No VecNormalize stats found at {args.source_vecnormalize}; "
                "initializing new normalization stats."
            )
            env = VecNormalize(env, norm_obs=True, norm_reward=False)

        print(f"Loading PPO model from {source_model_zip}")
        model = PPO.load(
            source_model_zip, tensorboard_log=args.resolved_tensorboard_log
        )
        assert_model_env_spaces_match(model, env)
        print(
            "Space compatibility OK: "
            f"observation={describe_space(env.observation_space)}, "
            f"action={describe_space(env.action_space)}"
        )
        model.set_env(env)

        set_model_learning_rate(model, args.learning_rate)
        print("Starting fine-tuning with reset_num_timesteps=False")
        training_started = True
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            tb_log_name=args.resolved_tb_log_name,
        )
        training_completed = True
    except KeyboardInterrupt:
        interrupted = True
        print("Transfer fine-tuning interrupted by user. Saving current state.")
    finally:
        end_time = datetime.now().isoformat(timespec="seconds")
        if env is not None and model is not None and training_started:
            os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(args.output_vecnormalize) or ".", exist_ok=True)
            print(f"Saving fine-tuned model to {output_model_zip}")
            model.save(args.output_model)
            print(f"Saving fine-tuned VecNormalize stats to {args.output_vecnormalize}")
            env.save(args.output_vecnormalize)

        if env is not None:
            print("Calling env.close()")
            env.close()

        if training_completed:
            phase = "completed"
        elif interrupted:
            phase = "interrupted"
        elif training_started:
            phase = "failed"
        else:
            phase = "not_started"
        write_json(
            config_path,
            transfer_config(
                args, robot_ids, experiment_dir, phase, start_time, end_time
            ),
        )
        print(f"Wrote transfer config to {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an existing PPO wheelchair model on a new Webots map."
    )
    parser.add_argument(
        "-w",
        "--world",
        default=DEFAULT_WORLD,
        help=(
            "World .wbt path used for this transfer run. The script does not launch "
            "Webots; open this world manually first."
        ),
    )
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Existing PPO model base path or .zip path to fine-tune from.",
    )
    parser.add_argument(
        "--source-vecnormalize",
        default=DEFAULT_SOURCE_VECNORMALIZE,
        help="Existing VecNormalize .pkl path to reuse if it exists.",
    )
    parser.add_argument(
        "--output-model",
        default=DEFAULT_OUTPUT_MODEL,
        help="New PPO model base path or .zip path for the fine-tuned model.",
    )
    parser.add_argument(
        "--output-vecnormalize",
        default=DEFAULT_OUTPUT_VECNORMALIZE,
        help="New VecNormalize .pkl path for the fine-tuned stats.",
    )
    parser.add_argument(
        "--experiment-name",
        default="transfer_finetune_lidar360",
        help="Name used in the structured experiment folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base directory for structured experiment outputs. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "-n",
        "--robots",
        type=int,
        default=N_ROBOTS,
        help=f"Number of robot clients to train with. Default: {N_ROBOTS}",
    )
    parser.add_argument(
        "--lidar-dim",
        type=int,
        default=LIDAR_DIM,
        help=f"LiDAR dimension used by the environment. Default: {LIDAR_DIM}",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=TRAIN_STEPS,
        help=f"Number of fine-tuning timesteps. Default: {TRAIN_STEPS}",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Fine-tuning learning rate. Default: 1e-5",
    )
    parser.add_argument(
        "--tensorboard-log",
        default=DEFAULT_TENSORBOARD_LOG,
        help=(
            "TensorBoard log directory passed to PPO. Default: "
            "<experiment>/training/tensorboard"
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=["default", "transfer_finetune", "transfer_goal_direction"],
        default=DEFAULT_REWARD_MODE,
        help=f"Reward mode for transfer environments. Default: {DEFAULT_REWARD_MODE}",
    )
    parser.add_argument(
        "--straight-bonus",
        type=float,
        default=STRAIGHT_BONUS_CLEAR,
        help=f"Forward-action bonus when front is clear. Default: {STRAIGHT_BONUS_CLEAR}",
    )
    parser.add_argument(
        "--soft-forward-bonus",
        type=float,
        default=SOFT_FORWARD_BONUS_CLEAR,
        help=(
            "Forward-left/right bonus when front is clear. "
            f"Default: {SOFT_FORWARD_BONUS_CLEAR}"
        ),
    )
    parser.add_argument(
        "--turn-penalty-clear",
        type=float,
        default=TURN_PENALTY_CLEAR,
        help=f"Left/right penalty when front is clear. Default: {TURN_PENALTY_CLEAR}",
    )
    parser.add_argument(
        "--stop-penalty-clear",
        type=float,
        default=STOP_PENALTY_CLEAR,
        help=f"Stop penalty when front is clear. Default: {STOP_PENALTY_CLEAR}",
    )
    parser.add_argument(
        "--oscillation-penalty",
        type=float,
        default=OSCILLATION_PENALTY,
        help=f"Penalty for repeated turn-ish actions. Default: {OSCILLATION_PENALTY}",
    )
    parser.add_argument(
        "--strong-oscillation-penalty",
        type=float,
        default=STRONG_OSCILLATION_PENALTY,
        help=(
            "Extra penalty for strict left-right-left-right loops. "
            f"Default: {STRONG_OSCILLATION_PENALTY}"
        ),
    )
    parser.add_argument(
        "--non-forward-clear-penalty",
        type=float,
        default=NON_FORWARD_CLEAR_PENALTY,
        help=(
            "Penalty when front is clear and recent actions rarely choose forward. "
            f"Default: {NON_FORWARD_CLEAR_PENALTY}"
        ),
    )
    parser.add_argument(
        "--clear-front-threshold",
        type=float,
        default=CLEAR_FRONT_THRESHOLD,
        help=(
            "Front clearance threshold for straight-line shaping. "
            f"Default: {CLEAR_FRONT_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--danger-front-threshold",
        type=float,
        default=DANGER_FRONT_THRESHOLD,
        help=(
            "Front clearance below which transfer shaping is disabled. "
            f"Default: {DANGER_FRONT_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--goal-direction-progress-weight",
        type=float,
        default=GOAL_DIRECTION_PROGRESS_WEIGHT,
        help=(
            "Weight applied to progress along the initial forward direction. "
            f"Default: {GOAL_DIRECTION_PROGRESS_WEIGHT}"
        ),
    )
    parser.add_argument(
        "--goal-direction-progress-clip",
        type=float,
        default=GOAL_DIRECTION_PROGRESS_CLIP,
        help=(
            "Absolute clip for goal-direction progress reward. "
            f"Default: {GOAL_DIRECTION_PROGRESS_CLIP}"
        ),
    )
    parser.add_argument(
        "--goal-direction-backward-penalty",
        type=float,
        default=GOAL_DIRECTION_BACKWARD_PENALTY,
        help=(
            "Penalty when progress along the initial direction is negative. "
            f"Default: {GOAL_DIRECTION_BACKWARD_PENALTY}"
        ),
    )
    parser.add_argument(
        "--goal-direction-loop-penalty",
        type=float,
        default=GOAL_DIRECTION_LOOP_PENALTY,
        help=(
            "Penalty for too little recent progress along the initial direction. "
            f"Default: {GOAL_DIRECTION_LOOP_PENALTY}"
        ),
    )
    parser.add_argument(
        "--goal-direction-loop-window",
        type=int,
        default=GOAL_DIRECTION_LOOP_WINDOW,
        help=(
            "Number of progress samples used for loop/stall detection. "
            f"Default: {GOAL_DIRECTION_LOOP_WINDOW}"
        ),
    )
    parser.add_argument(
        "--goal-direction-min-recent-progress",
        type=float,
        default=GOAL_DIRECTION_MIN_RECENT_PROGRESS,
        help=(
            "Minimum net progress over the loop window before applying loop penalty. "
            f"Default: {GOAL_DIRECTION_MIN_RECENT_PROGRESS}"
        ),
    )
    parser.add_argument(
        "--goal-direction-clear-front-threshold",
        type=float,
        default=GOAL_DIRECTION_CLEAR_FRONT_THRESHOLD,
        help=(
            "Front clearance threshold for heading alignment penalty. "
            f"Default: {GOAL_DIRECTION_CLEAR_FRONT_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--goal-direction-heading-weight",
        type=float,
        default=GOAL_DIRECTION_HEADING_WEIGHT,
        help=(
            "Weight for heading drift penalty when the front path is clear. "
            f"Default: {GOAL_DIRECTION_HEADING_WEIGHT}"
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing fine-tuned model/stat output file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate paths and robot IDs, then exit before importing RL libraries or "
            "creating the environment."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        fine_tune(parse_args())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
