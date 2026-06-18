import argparse
import csv
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime


LIDAR_DIM = 360
N_ROBOTS = 9
MAX_TRAIN_TIMESTEPS = 10_000_000
EVAL_EPISODES = 50
DEFAULT_TRAIN_WORLD = "worlds/smart-wheelchairs-transfer-finetune.wbt"
DEFAULT_EVAL_WORLD = "worlds/smart-wheelchairs-realworld-test.wbt"
DEFAULT_OUTPUT_ROOT = "outputs/pipeline"
DEFAULT_SOURCE_MODEL = f"models/ppo_wheelchair_lidar{LIDAR_DIM}_human"
DEFAULT_SOURCE_VECNORMALIZE = f"models/vecnormalize_lidar{LIDAR_DIM}_human.pkl"
DEFAULT_LEARNING_RATE_TRANSFER = 1e-5
DEFAULT_LEARNING_RATE_SCRATCH = 5e-5
TRAINABLE_EXPERIMENTS = [
    "scratch",
    "transfer_finetune",
    "transfer_goal_direction",
]
EVAL_ONLY_EXPERIMENTS = ["base_easy_model"]
ALL_EXPERIMENTS = EVAL_ONLY_EXPERIMENTS + TRAINABLE_EXPERIMENTS
PHASES = ["check", "train", "eval", "report", "all"]
EARLY_STOP_METRICS = ["mean_reward", "success_rate", "avg_time_to_goal"]
ACTION_LABELS = [
    "forward",
    "forward_left",
    "forward_right",
    "left",
    "right",
    "stop",
]


class WebotsLaunchError(RuntimeError):
    def __init__(self, message, attempt=None):
        super().__init__(message)
        self.attempt = attempt


def with_zip_suffix(path):
    return path if path.endswith(".zip") else path + ".zip"


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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


def parse_experiments(value):
    selected = []
    for raw_name in value.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in TRAINABLE_EXPERIMENTS:
            raise argparse.ArgumentTypeError(
                f"unknown experiment '{name}'. Choose from: "
                + ", ".join(TRAINABLE_EXPERIMENTS)
            )
        if name not in selected:
            selected.append(name)

    if not selected:
        raise argparse.ArgumentTypeError("at least one experiment must be selected")
    return selected


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 6 scaffold for the PPO transfer comparison pipeline. "
            "This stage validates configuration, can launch Webots, and can "
            "train/evaluate/report selected models with optional early stopping."
        )
    )
    parser.add_argument("--train-world", default=DEFAULT_TRAIN_WORLD)
    parser.add_argument("--eval-world", default=DEFAULT_EVAL_WORLD)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-timesteps", type=int, default=MAX_TRAIN_TIMESTEPS)
    parser.add_argument("--robots", type=int, default=N_ROBOTS)
    parser.add_argument("--lidar-dim", type=int, default=LIDAR_DIM)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-vecnormalize", default=DEFAULT_SOURCE_VECNORMALIZE)
    parser.add_argument(
        "--learning-rate-transfer",
        type=float,
        default=DEFAULT_LEARNING_RATE_TRANSFER,
    )
    parser.add_argument(
        "--learning-rate-scratch",
        type=float,
        default=DEFAULT_LEARNING_RATE_SCRATCH,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress-log-freq",
        type=int,
        default=10_000,
        help="Timesteps/steps between progress log entries. Default: 10000.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Alias for --phase check. Does not instantiate the environment.",
    )
    parser.add_argument(
        "--check-env-spaces",
        action="store_true",
        help=(
            "Accepted for later stages. Real env-space validation may block unless "
            "Webots robot clients are running."
        ),
    )
    parser.add_argument(
        "--experiments",
        type=parse_experiments,
        default=list(TRAINABLE_EXPERIMENTS),
        help=(
            "Comma-separated trainable experiments. Default: "
            "scratch,transfer_finetune,transfer_goal_direction"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help=(
            "Pipeline phase to run. Stage 6 implements static checks, optional "
            "Webots launch/cleanup, training, evaluation, reporting, and optional "
            "early stopping."
        ),
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Existing outputs/pipeline run directory to reuse for later phases.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--eval-checkpoint",
        choices=["final", "best"],
        default="final",
        help="Checkpoint to evaluate for trained models. Default: final.",
    )

    parser.add_argument("--launch-webots", action="store_true")
    parser.add_argument("--webots-bin", default="webots")
    parser.add_argument("--webots-mode", default="fast")
    parser.add_argument("--webots-no-rendering", action="store_true")
    parser.add_argument("--webots-startup-wait", type=float, default=10.0)

    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument(
        "--early-stop-metric",
        choices=EARLY_STOP_METRICS,
        default="mean_reward",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0)
    parser.add_argument("--early-stop-min-timesteps", type=int, default=200_000)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=10)

    return parser


def resolve_phase(args):
    if args.check_only:
        return "check"
    return args.phase


def make_run_dir(args, phase):
    if args.resume_run:
        run_dir = args.resume_run
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"--resume-run does not exist: {run_dir}")
        return run_dir, True

    base_run_dir = os.path.join(
        args.output_root,
        f"{timestamp()}_transfer_comparison",
    )
    run_dir = base_run_dir
    if os.path.exists(run_dir) and not args.overwrite:
        for index in range(1, 100):
            candidate = f"{base_run_dir}_{index:02d}"
            if not os.path.exists(candidate):
                run_dir = candidate
                break
        else:
            raise FileExistsError(
                f"Output run directory already exists: {base_run_dir}. "
                "Pass --overwrite to reuse it."
            )

    os.makedirs(run_dir, exist_ok=True)
    return run_dir, False


def create_pipeline_folders(run_dir):
    for folder in ["logs", "models", "eval", "plots", "reports"]:
        os.makedirs(os.path.join(run_dir, folder), exist_ok=True)

    for experiment in ALL_EXPERIMENTS:
        os.makedirs(os.path.join(run_dir, experiment), exist_ok=True)
        for folder in ["eval", "eval/trajectories"]:
            os.makedirs(os.path.join(run_dir, experiment, folder), exist_ok=True)
        if experiment in TRAINABLE_EXPERIMENTS:
            for folder in ["model", "vecnormalize", "logs", "logs/tensorboard"]:
                os.makedirs(os.path.join(run_dir, experiment, folder), exist_ok=True)


def experiment_paths(run_dir, experiment):
    experiment_dir = os.path.join(run_dir, experiment)
    return {
        "experiment_dir": experiment_dir,
        "model": os.path.join(experiment_dir, "model", "final_model.zip"),
        "model_base": os.path.join(experiment_dir, "model", "final_model"),
        "best_model": os.path.join(experiment_dir, "model", "best_model.zip"),
        "best_model_base": os.path.join(experiment_dir, "model", "best_model"),
        "failed_model_base": os.path.join(experiment_dir, "model", "failed_model"),
        "failed_model": os.path.join(experiment_dir, "model", "failed_model.zip"),
        "vecnormalize": os.path.join(
            experiment_dir, "vecnormalize", "final_vecnormalize.pkl"
        ),
        "best_vecnormalize": os.path.join(
            experiment_dir, "vecnormalize", "best_vecnormalize.pkl"
        ),
        "failed_vecnormalize": os.path.join(
            experiment_dir, "vecnormalize", "failed_vecnormalize.pkl"
        ),
        "tensorboard": os.path.join(experiment_dir, "logs", "tensorboard"),
        "progress_log": os.path.join(experiment_dir, "logs", "progress.log"),
        "training_metrics": os.path.join(experiment_dir, "training_metrics.json"),
        "early_stopping_csv": os.path.join(experiment_dir, "early_stopping.csv"),
        "early_stopping_json": os.path.join(experiment_dir, "early_stopping.json"),
        "eval_dir": os.path.join(experiment_dir, "eval"),
        "eval_metrics": os.path.join(experiment_dir, "eval", "metrics.json"),
        "eval_episode_metrics": os.path.join(
            experiment_dir, "eval", "episode_metrics.csv"
        ),
        "eval_action_distribution": os.path.join(
            experiment_dir, "eval", "action_distribution.csv"
        ),
        "eval_trajectories": os.path.join(experiment_dir, "eval", "trajectories"),
        "eval_error_log": os.path.join(experiment_dir, "eval", "error.log"),
        "error_log": os.path.join(experiment_dir, "error.log"),
    }


def validate_world(world_path, requested_robots, label, fatal_robot_mismatch):
    result = {
        "label": label,
        "path": world_path,
        "exists": os.path.exists(world_path),
        "robot_ids": [],
        "requested_robots": requested_robots,
        "status": "ok",
        "warnings": [],
        "errors": [],
    }

    if not result["exists"]:
        result["status"] = "error"
        result["errors"].append(f"{label} world does not exist: {world_path}")
        return result

    try:
        robot_ids = parse_robot_client_ids(world_path)
    except OSError as exc:
        result["status"] = "error"
        result["errors"].append(f"could not read {label} world: {exc}")
        return result

    result["robot_ids"] = robot_ids
    expected = list(range(requested_robots))
    missing = [robot_id for robot_id in expected if robot_id not in robot_ids]
    if not robot_ids:
        result["status"] = "error"
        result["errors"].append(f"no robot_client controllerArgs found in {world_path}")
    elif missing:
        message = (
            f"{label} world has robot client IDs {robot_ids}, but --robots "
            f"{requested_robots} expects IDs {expected}. Missing: {missing}"
        )
        if fatal_robot_mismatch:
            result["status"] = "error"
            result["errors"].append(message)
        else:
            result["status"] = "warning"
            result["warnings"].append(message)

    return result


def validate_paths(args):
    validations = {
        "errors": [],
        "warnings": [],
        "paths": {},
        "worlds": {},
    }

    source_model_zip = with_zip_suffix(args.source_model)
    validations["paths"]["source_model"] = {
        "path": source_model_zip,
        "exists": os.path.exists(source_model_zip),
    }
    if not os.path.exists(source_model_zip):
        validations["errors"].append(f"source model does not exist: {source_model_zip}")

    validations["paths"]["source_vecnormalize"] = {
        "path": args.source_vecnormalize,
        "exists": os.path.exists(args.source_vecnormalize),
    }
    if not os.path.exists(args.source_vecnormalize):
        validations["warnings"].append(
            f"source VecNormalize file not found: {args.source_vecnormalize}"
        )

    validations["worlds"]["train"] = validate_world(
        args.train_world,
        args.robots,
        "training",
        fatal_robot_mismatch=True,
    )
    validations["worlds"]["eval"] = validate_world(
        args.eval_world,
        args.robots,
        "evaluation",
        fatal_robot_mismatch=False,
    )

    for world_result in validations["worlds"].values():
        validations["errors"].extend(world_result["errors"])
        validations["warnings"].extend(world_result["warnings"])

    if args.lidar_dim != LIDAR_DIM:
        validations["warnings"].append(
            f"--lidar-dim is {args.lidar_dim}; the base model expects {LIDAR_DIM}."
        )

    if args.check_env_spaces:
        validations["warnings"].append(
            "--check-env-spaces is accepted in Stage 6 but not executed. "
            "A later stage should warn before instantiating WheelchairEnv because "
            "it may block without Webots."
        )

    return validations


def experiment_matrix(args):
    matrix = {
        "base_easy_model": {
            "type": "evaluation_only",
            "trained_or_eval_only": "eval_only",
            "train_world": None,
            "eval_world": args.eval_world,
            "source_model": with_zip_suffix(args.source_model),
            "source_vecnormalize": args.source_vecnormalize,
            "reward_mode": "base_model_original",
        }
    }

    for experiment in TRAINABLE_EXPERIMENTS:
        matrix[experiment] = {
            "type": "trainable",
            "selected_for_training": experiment in args.experiments,
            "trained_or_eval_only": "trained",
            "train_world": args.train_world,
            "eval_world": args.eval_world,
            "reward_mode": "default" if experiment == "scratch" else experiment,
        }

    return matrix


def planned_webots_commands(args):
    train_cmd = build_webots_command(args, args.train_world)
    eval_cmd = build_webots_command(args, args.eval_world)
    return {"train": train_cmd, "eval": eval_cmd}


def build_webots_command(args, world_path):
    command = [args.webots_bin, f"--mode={args.webots_mode}"]
    if args.webots_no_rendering:
        command.append("--no-rendering")
    command.append(world_path)
    return command


def package_versions():
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            versions["git_commit"] = completed.stdout.strip()
    except OSError:
        versions["git_commit"] = None
    return versions


def append_config_event(config_path, config):
    write_json(config_path, config)


def pipeline_progress_log_path(config):
    return os.path.join(config["run_dir"], "logs", "progress.log")


def experiment_progress_log_path(config, experiment):
    return experiment_paths(config["run_dir"], experiment)["progress_log"]


def progress_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_progress(config, message, experiment=None, level="INFO"):
    if level == "INFO":
        line = f"[{progress_timestamp()}] {message}"
    else:
        line = f"[{progress_timestamp()}] {level} {message}"

    root_path = pipeline_progress_log_path(config)
    os.makedirs(os.path.dirname(root_path), exist_ok=True)
    with open(root_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if experiment is not None:
        experiment_path = experiment_progress_log_path(config, experiment)
        os.makedirs(os.path.dirname(experiment_path), exist_ok=True)
        with open(experiment_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    print(line, flush=True)


def safe_metric_value(value):
    if value is None:
        return "NA"
    try:
        if isinstance(value, float):
            return f"{value:.4g}"
    except TypeError:
        pass
    return str(value)


def launch_webots(args, run_dir, phase_label, world_path, config=None, experiment=None):
    log_path = os.path.join(run_dir, "logs", f"webots_{phase_label}.log")
    command = build_webots_command(args, world_path)
    attempt = {
        "phase": phase_label,
        "world": world_path,
        "command": command,
        "log_path": log_path,
        "startup_wait": args.webots_startup_wait,
        "started": False,
        "stopped": False,
        "status": "pending",
        "pid": None,
        "returncode": None,
        "error": None,
    }

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        if config is not None:
            log_progress(
                config,
                "WEBOTS "
                f"{phase_label.upper()} launch command={' '.join(command)} "
                f"world={world_path} log={log_path}",
                experiment=experiment,
            )
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        log_file.close()
        attempt["status"] = "failed"
        attempt["error"] = (
            f"Webots executable not found while running command: {command}"
        )
        if config is not None:
            log_progress(config, attempt["error"], experiment=experiment, level="ERROR")
        raise WebotsLaunchError(attempt["error"], attempt) from exc
    except OSError as exc:
        log_file.close()
        attempt["status"] = "failed"
        attempt["error"] = f"failed to launch Webots command {command}: {exc}"
        if config is not None:
            log_progress(config, attempt["error"], experiment=experiment, level="ERROR")
        raise WebotsLaunchError(attempt["error"], attempt) from exc

    attempt["started"] = True
    attempt["pid"] = process.pid
    attempt["status"] = "started"
    if config is not None:
        log_progress(
            config,
            f"WEBOTS {phase_label.upper()} launched world={world_path} pid={process.pid}",
            experiment=experiment,
        )
    if args.webots_startup_wait > 0:
        time.sleep(args.webots_startup_wait)

    returncode = process.poll()
    if returncode is not None:
        log_file.close()
        attempt["returncode"] = returncode
        attempt["status"] = "failed"
        attempt["error"] = (
            "Webots exited during startup wait. "
            f"See log: {log_path}"
        )
        if config is not None:
            log_progress(
                config,
                "WEBOTS "
                f"{phase_label.upper()} startup failed returncode={returncode} "
                f"log={log_path}",
                experiment=experiment,
                level="ERROR",
            )
        raise WebotsLaunchError(attempt["error"], attempt)

    attempt["status"] = "running"
    if config is not None:
        log_progress(
            config,
            f"WEBOTS {phase_label.upper()} startup verified pid={process.pid}",
            experiment=experiment,
        )
    return process, log_file, attempt


def stop_webots(process, log_file, attempt, timeout=10, config=None, experiment=None):
    if process.poll() is None:
        process.terminate()
        try:
            attempt["returncode"] = process.wait(timeout=timeout)
            attempt["status"] = "stopped"
        except subprocess.TimeoutExpired:
            process.kill()
            attempt["returncode"] = process.wait(timeout=timeout)
            attempt["status"] = "killed"
            attempt["error"] = "Webots did not terminate cleanly; killed process."
    else:
        attempt["returncode"] = process.returncode
        if attempt["status"] == "running":
            attempt["status"] = "exited_before_stop"

    attempt["stopped"] = True
    log_file.close()
    if config is not None:
        level = "ERROR" if attempt.get("status") == "killed" else "INFO"
        log_progress(
            config,
            "WEBOTS "
            f"{attempt['phase'].upper()} stop status={attempt['status']} "
            f"returncode={attempt.get('returncode')} log={attempt['log_path']}",
            experiment=experiment,
            level=level,
        )
    return attempt


def manual_webots_message(args, phase):
    if phase == "train":
        print(f"Manual mode: open training world before training: {args.train_world}")
    elif phase == "eval":
        print(f"Manual mode: open evaluation world before evaluation: {args.eval_world}")
    elif phase == "all":
        print("Manual mode:")
        print(f"  1. Open training world first: {args.train_world}")
        print(f"  2. Close it before evaluation.")
        print(f"  3. Open evaluation world next: {args.eval_world}")


def webots_phases_for_pipeline_phase(phase):
    if phase == "train":
        return [("train", "train_world")]
    if phase == "eval":
        return [("eval", "eval_world")]
    if phase == "all":
        return [("train", "train_world"), ("eval", "eval_world")]
    return []


def smoke_test_webots(args, config, config_path):
    phase = config["phase"]
    if not args.launch_webots:
        manual_webots_message(args, phase)
        return 0

    for webots_phase, world_key in webots_phases_for_pipeline_phase(phase):
        world_path = config["paths"][world_key]
        print(f"Launching Webots {webots_phase} world: {world_path}")
        attempt = None
        process = None
        log_file = None
        try:
            process, log_file, attempt = launch_webots(
                args,
                config["run_dir"],
                webots_phase,
                world_path,
                config=config,
            )
            config["webots"]["launch_attempts"].append(attempt)
            append_config_event(config_path, config)
            print(
                f"Webots {webots_phase} startup verified; stopping process "
                f"{process.pid}."
            )
        except WebotsLaunchError as exc:
            attempt = exc.attempt or attempt
            if attempt is None:
                attempt = {
                    "phase": webots_phase,
                    "world": world_path,
                    "command": build_webots_command(args, world_path),
                    "log_path": os.path.join(
                        config["run_dir"], "logs", f"webots_{webots_phase}.log"
                    ),
                    "startup_wait": args.webots_startup_wait,
                    "started": False,
                    "stopped": False,
                    "status": "failed",
                    "pid": None,
                    "returncode": None,
                    "error": str(exc),
                }
            elif attempt not in config["webots"]["launch_attempts"]:
                config["webots"]["launch_attempts"].append(attempt)
            append_config_event(config_path, config)
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        finally:
            if process is not None and log_file is not None and attempt is not None:
                stop_webots(process, log_file, attempt, config=config)
                append_config_event(config_path, config)
                print(
                    f"Stopped Webots {webots_phase}; log: {attempt['log_path']}"
                )

    return 0


def patch_numpy_pickle_modules():
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
    mismatches = []
    if not spaces_match(model.observation_space, env.observation_space):
        mismatches.append(
            "observation space mismatch: "
            f"model={describe_space(model.observation_space)}, "
            f"env={describe_space(env.observation_space)}"
        )
    if not spaces_match(model.action_space, env.action_space):
        mismatches.append(
            "action space mismatch: "
            f"model={describe_space(model.action_space)}, "
            f"env={describe_space(env.action_space)}"
        )
    if mismatches:
        raise ValueError(
            "Loaded model is not compatible with the training environment:\n"
            + "\n".join(f"  - {mismatch}" for mismatch in mismatches)
        )


def set_model_learning_rate(model, learning_rate):
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _: learning_rate
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = learning_rate


def reward_mode_for_experiment(experiment):
    return "default" if experiment == "scratch" else experiment


def make_training_env(env_id, args, reward_mode):
    from stable_baselines3.common.monitor import Monitor
    from wheelchair_env import WheelchairEnv

    def _init():
        return Monitor(
            WheelchairEnv(
                env_id,
                lidar_dim=args.lidar_dim,
                reward_mode=reward_mode,
            )
        )

    return _init


def make_training_record(run_dir, experiment, args):
    paths = experiment_paths(run_dir, experiment)
    return {
        "status": "pending",
        "experiment": experiment,
        "reward_mode": reward_mode_for_experiment(experiment),
        "start_time": None,
        "end_time": None,
        "requested_timesteps": args.max_timesteps,
        "actual_timesteps": 0,
        "wall_clock_seconds": 0.0,
        "model_path": paths["model"],
        "vecnormalize_path": paths["vecnormalize"],
        "tensorboard_log": paths["tensorboard"],
        "progress_log": paths["progress_log"],
        "training_metrics": paths["training_metrics"],
        "error_log": None,
        "interrupted": False,
        "early_stopping": {
            "enabled": args.early_stop,
            "best_metric_value": None,
            "actual_stop_timestep": 0,
            "early_stopped": False,
            "stop_reason": None,
            "best_model_path": paths["best_model"],
            "best_vecnormalize_path": paths["best_vecnormalize"],
        },
    }


def ensure_training_records(config, args):
    training = config.setdefault("training", {})
    records = training.setdefault("experiments", {})
    for experiment in TRAINABLE_EXPERIMENTS:
        records.setdefault(
            experiment,
            make_training_record(config["run_dir"], experiment, args),
        )
    return records


def save_training_metrics(record):
    write_json(record["training_metrics"], record)


def write_error_log(path, exc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{exc.__class__.__name__}: {exc}\n\n")
        f.write(traceback.format_exc())


def early_stop_metric_direction(metric_name):
    return "minimize" if metric_name == "avg_time_to_goal" else "maximize"


def early_stop_improved(metric_name, value, best_value, min_delta):
    if value is None:
        return False
    if best_value is None:
        return True
    if early_stop_metric_direction(metric_name) == "minimize":
        return value <= best_value - min_delta
    return value >= best_value + min_delta


def write_early_stopping_outputs(paths, rows):
    fieldnames = [
        "timestep",
        "metric_name",
        "metric_value",
        "best_metric_value",
        "improved",
        "patience_counter",
        "early_stop_allowed",
        "stopped",
        "stop_reason",
    ]
    with open(paths["early_stopping_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(paths["early_stopping_json"], rows)


def early_stop_evaluate(model, env, episodes):
    previous_training = getattr(env, "training", False)
    env.training = False
    env.norm_reward = False
    rewards = []
    successes = 0
    time_to_goal = []
    completed = 0
    current_rewards = [0.0 for _ in range(env.num_envs)]
    current_steps = [0 for _ in range(env.num_envs)]
    safety_steps = max(1000, episodes * 20000)
    steps = 0
    try:
        obs = env.reset()
        while completed < episodes and steps < safety_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_values, dones, infos = env.step(action)
            steps += 1
            for index in range(env.num_envs):
                current_rewards[index] += float(reward_values[index])
                current_steps[index] += 1
            for index, done in enumerate(dones):
                if not done:
                    continue
                info = infos[index]
                completed += 1
                rewards.append(current_rewards[index])
                if info.get("is_success", False):
                    successes += 1
                    time_to_goal.append(float(info.get("time_step", current_steps[index])))
                current_rewards[index] = 0.0
                current_steps[index] = 0
                if completed >= episodes:
                    break
    finally:
        env.training = previous_training

    return {
        "mean_reward": mean_value(rewards) if rewards else None,
        "success_rate": 100 * successes / completed if completed else None,
        "avg_time_to_goal": mean_value(time_to_goal) if time_to_goal else None,
        "completed_episodes": completed,
        "completed_steps": steps,
    }


def save_failed_checkpoint(model, env, paths):
    if model is not None:
        os.makedirs(os.path.dirname(paths["failed_model_base"]), exist_ok=True)
        model.save(paths["failed_model_base"])
    if env is not None:
        os.makedirs(os.path.dirname(paths["failed_vecnormalize"]), exist_ok=True)
        env.save(paths["failed_vecnormalize"])


def make_training_progress_callback(
    args,
    config,
    experiment,
    reward_mode,
    start_monotonic,
):
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressLogCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_logged_timesteps = 0

        def _on_step(self):
            current_timesteps = int(getattr(self.model, "num_timesteps", 0))
            interval = max(1, int(args.progress_log_freq))
            if current_timesteps - self.last_logged_timesteps < interval:
                return True

            logger_values = getattr(self.logger, "name_to_value", {}) or {}
            elapsed = round(time.monotonic() - start_monotonic, 1)
            log_progress(
                config,
                "TRAIN PROGRESS "
                f"experiment={experiment} "
                f"timesteps={current_timesteps}/{args.max_timesteps} "
                f"elapsed={elapsed}s "
                f"fps={safe_metric_value(logger_values.get('time/fps'))} "
                "ep_rew_mean="
                f"{safe_metric_value(logger_values.get('rollout/ep_rew_mean'))} "
                "ep_len_mean="
                f"{safe_metric_value(logger_values.get('rollout/ep_len_mean'))} "
                f"reward_mode={reward_mode}",
                experiment=experiment,
            )
            self.last_logged_timesteps = current_timesteps
            return True

    return ProgressLogCallback()


def learn_with_optional_early_stopping(
    args,
    model,
    env,
    paths,
    record,
    reset_num_timesteps,
    tb_log_name,
    config,
    experiment,
    reward_mode,
    progress_callback,
):
    if not args.early_stop:
        log_progress(
            config,
            "EARLY STOP disabled "
            f"experiment={experiment} default_training=model.learn(total_timesteps)",
            experiment=experiment,
        )
        model.learn(
            total_timesteps=args.max_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            callback=progress_callback,
        )
        record["early_stopping"]["actual_stop_timestep"] = int(
            getattr(model, "num_timesteps", 0)
        )
        record["early_stopping"]["stop_reason"] = "max_timesteps_reached"
        return

    rows = []
    best_metric_value = None
    patience_counter = 0
    completed_evaluations = 0
    remaining = args.max_timesteps
    first_chunk = True
    stop_reason = "max_timesteps_reached"
    log_progress(
        config,
        "EARLY STOP enabled "
        f"experiment={experiment} metric={args.early_stop_metric} "
        f"direction={early_stop_metric_direction(args.early_stop_metric)} "
        f"eval_freq={args.eval_freq} patience={args.early_stop_patience} "
        f"min_timesteps={args.early_stop_min_timesteps}",
        experiment=experiment,
    )
    while remaining > 0:
        chunk_timesteps = min(args.eval_freq, remaining)
        model.learn(
            total_timesteps=chunk_timesteps,
            reset_num_timesteps=reset_num_timesteps if first_chunk else False,
            tb_log_name=tb_log_name,
            callback=progress_callback,
        )
        first_chunk = False
        remaining -= chunk_timesteps
        current_timestep = int(getattr(model, "num_timesteps", 0))
        try:
            eval_metrics = early_stop_evaluate(
                model, env, args.early_stop_eval_episodes
            )
        except Exception:
            save_failed_checkpoint(model, env, paths)
            log_progress(
                config,
                "EARLY STOP EVAL failed "
                f"experiment={experiment} timestep={current_timestep} "
                f"failed_model={paths['failed_model']} "
                f"failed_vecnormalize={paths['failed_vecnormalize']}",
                experiment=experiment,
                level="ERROR",
            )
            raise

        metric_value = eval_metrics.get(args.early_stop_metric)
        improved = early_stop_improved(
            args.early_stop_metric,
            metric_value,
            best_metric_value,
            args.early_stop_min_delta,
        )
        completed_evaluations += 1
        if improved:
            best_metric_value = metric_value
            patience_counter = 0
            model.save(paths["best_model_base"])
            env.save(paths["best_vecnormalize"])
            log_progress(
                config,
                "EARLY STOP BEST "
                f"experiment={experiment} timestep={current_timestep} "
                f"metric={args.early_stop_metric} "
                f"value={safe_metric_value(metric_value)} "
                f"model={paths['best_model']} "
                f"vecnormalize={paths['best_vecnormalize']}",
                experiment=experiment,
            )
        else:
            patience_counter += 1

        early_stop_allowed = (
            current_timestep >= args.early_stop_min_timesteps
            and completed_evaluations >= args.early_stop_patience
        )
        stopped = early_stop_allowed and patience_counter >= args.early_stop_patience
        if stopped:
            stop_reason = "patience_exhausted"

        row = {
            "timestep": current_timestep,
            "metric_name": args.early_stop_metric,
            "metric_value": metric_value,
            "best_metric_value": best_metric_value,
            "improved": improved,
            "patience_counter": patience_counter,
            "early_stop_allowed": early_stop_allowed,
            "stopped": stopped,
            "stop_reason": stop_reason if stopped else "",
        }
        rows.append(row)
        write_early_stopping_outputs(paths, rows)
        log_progress(
            config,
            "EARLY STOP EVAL "
            f"experiment={experiment} timestep={current_timestep} "
            f"metric_name={args.early_stop_metric} "
            f"metric_value={safe_metric_value(metric_value)} "
            f"best_metric_value={safe_metric_value(best_metric_value)} "
            f"improved={improved} patience_counter={patience_counter} "
            f"early_stop_allowed={early_stop_allowed} stopped={stopped} "
            f"stop_reason={row['stop_reason'] or 'none'}",
            experiment=experiment,
        )
        if stopped:
            break

    record["early_stopping"].update({
        "enabled": True,
        "best_metric_value": best_metric_value,
        "actual_stop_timestep": int(getattr(model, "num_timesteps", 0)),
        "early_stopped": stop_reason == "patience_exhausted",
        "stop_reason": stop_reason,
        "metric_direction": early_stop_metric_direction(args.early_stop_metric),
        "best_model_path": paths["best_model"],
        "best_vecnormalize_path": paths["best_vecnormalize"],
    })
    log_progress(
        config,
        "EARLY STOP END "
        f"experiment={experiment} stop_reason={stop_reason} "
        f"actual_timestep={record['early_stopping']['actual_stop_timestep']}",
        experiment=experiment,
    )


def create_scratch_model(args, env, tensorboard_log):
    from stable_baselines3 import PPO
    from manual_feature_extractor import ManualFeatureExtractor

    policy_kwargs = dict(
        features_extractor_class=ManualFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=8192,
        learning_rate=args.learning_rate_scratch,
        batch_size=1024,
        n_epochs=20,
        clip_range=0.1,
        ent_coef=0.01,
        device="cpu",
        tensorboard_log=tensorboard_log,
        seed=args.seed,
    )


def create_transfer_model(args, env, reward_mode, tensorboard_log):
    from stable_baselines3 import PPO

    source_model_zip = with_zip_suffix(args.source_model)
    print(f"Loading source PPO model for {reward_mode}: {source_model_zip}")
    model = PPO.load(source_model_zip, tensorboard_log=tensorboard_log)
    assert_model_env_spaces_match(model, env)
    model.set_env(env)
    set_model_learning_rate(model, args.learning_rate_transfer)
    if args.seed is not None and hasattr(model, "set_random_seed"):
        model.set_random_seed(args.seed)
    return model


def train_one_experiment(args, config, config_path, experiment):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    patch_numpy_pickle_modules()
    records = ensure_training_records(config, args)
    record = records[experiment]
    paths = experiment_paths(config["run_dir"], experiment)
    reward_mode = reward_mode_for_experiment(experiment)
    env = None
    model = None
    training_started = False
    start_monotonic = time.monotonic()
    record.update({
        "status": "running",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "end_time": None,
        "requested_timesteps": args.max_timesteps,
        "actual_timesteps": 0,
        "wall_clock_seconds": 0.0,
        "error_log": None,
        "interrupted": False,
    })
    append_config_event(config_path, config)
    save_training_metrics(record)
    log_progress(
        config,
        "EXPERIMENT START "
        f"experiment={experiment} phase=train reward_mode={reward_mode}",
        experiment=experiment,
    )
    log_progress(
        config,
        "TRAIN START "
        f"experiment={experiment} timesteps={args.max_timesteps} "
        f"reward_mode={reward_mode} world={config['paths']['train_world']} "
        f"progress_log_freq={args.progress_log_freq}",
        experiment=experiment,
    )

    try:
        print(f"Starting training experiment: {experiment}")
        env = DummyVecEnv(
            [
                make_training_env(robot_id, args, reward_mode)
                for robot_id in range(args.robots)
            ]
        )
        if experiment == "scratch":
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            model = create_scratch_model(args, env, paths["tensorboard"])
            reset_num_timesteps = True
        else:
            if os.path.exists(args.source_vecnormalize):
                print(f"Loading source VecNormalize: {args.source_vecnormalize}")
                env = VecNormalize.load(args.source_vecnormalize, env)
                env.training = True
                env.norm_reward = False
            else:
                print(
                    f"No source VecNormalize found at {args.source_vecnormalize}; "
                    "initializing new normalization stats."
                )
                env = VecNormalize(env, norm_obs=True, norm_reward=False)
            model = create_transfer_model(args, env, reward_mode, paths["tensorboard"])
            reset_num_timesteps = False

        training_started = True
        progress_callback = make_training_progress_callback(
            args,
            config,
            experiment,
            reward_mode,
            start_monotonic,
        )
        learn_with_optional_early_stopping(
            args,
            model,
            env,
            paths,
            record,
            reset_num_timesteps,
            f"ppo-lidar{args.lidar_dim}-{experiment}",
            config,
            experiment,
            reward_mode,
            progress_callback,
        )
        record["status"] = "completed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["interrupted"] = True
        save_failed_checkpoint(model, env, paths)
        log_progress(
            config,
            "TRAIN INTERRUPTED "
            f"experiment={experiment} failed_model={paths['failed_model']} "
            f"failed_vecnormalize={paths['failed_vecnormalize']}",
            experiment=experiment,
            level="ERROR",
        )
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error_log"] = paths["error_log"]
        save_failed_checkpoint(model, env, paths)
        write_error_log(paths["error_log"], exc)
        log_progress(
            config,
            "TRAIN ERROR "
            f"experiment={experiment} error={exc} error_log={paths['error_log']} "
            f"failed_model={paths['failed_model']} "
            f"failed_vecnormalize={paths['failed_vecnormalize']}",
            experiment=experiment,
            level="ERROR",
        )
        raise
    finally:
        record["end_time"] = datetime.now().isoformat(timespec="seconds")
        record["wall_clock_seconds"] = round(time.monotonic() - start_monotonic, 3)
        if model is not None:
            record["actual_timesteps"] = int(getattr(model, "num_timesteps", 0))
        if env is not None and model is not None and training_started:
            os.makedirs(os.path.dirname(paths["model_base"]), exist_ok=True)
            os.makedirs(os.path.dirname(paths["vecnormalize"]), exist_ok=True)
            print(f"Saving model to {paths['model']}")
            model.save(paths["model_base"])
            print(f"Saving VecNormalize to {paths['vecnormalize']}")
            env.save(paths["vecnormalize"])
            log_progress(
                config,
                "TRAIN SAVE "
                f"experiment={experiment} model={paths['model']} "
                f"vecnormalize={paths['vecnormalize']}",
                experiment=experiment,
            )
        if env is not None:
            print(f"Closing training environment for {experiment}")
            try:
                env.close()
            except Exception as close_exc:
                record.setdefault("warnings", []).append(
                    f"env.close() failed: {close_exc}"
                )
        save_training_metrics(record)
        append_config_event(config_path, config)

    log_progress(
        config,
        "TRAIN END "
        f"experiment={experiment} actual_timesteps={record['actual_timesteps']} "
        f"elapsed={record['wall_clock_seconds']}s model={paths['model']} "
        f"vecnormalize={paths['vecnormalize']}",
        experiment=experiment,
    )
    log_progress(
        config,
        f"EXPERIMENT END experiment={experiment} phase=train status={record['status']}",
        experiment=experiment,
    )
    print(f"Training experiment completed: {experiment}")


def selected_training_experiments(args):
    if args.skip_training:
        return []
    return list(args.experiments)


def run_training_experiments(args, config, config_path):
    ensure_training_records(config, args)
    experiments = selected_training_experiments(args)
    if not experiments:
        print("Training skipped by --skip-training or empty experiment selection.")
        log_progress(config, "TRAIN SKIP reason=skip_training_or_empty_selection")
        append_config_event(config_path, config)
        return 0

    for index, experiment in enumerate(experiments):
        process = None
        log_file = None
        attempt = None
        try:
            if args.launch_webots:
                label = "train" if len(experiments) == 1 else f"train_{experiment}"
                print(f"Launching Webots training world for {experiment}.")
                process, log_file, attempt = launch_webots(
                    args,
                    config["run_dir"],
                    label,
                    config["paths"]["train_world"],
                    config=config,
                    experiment=experiment,
                )
                config["webots"]["launch_attempts"].append(attempt)
                append_config_event(config_path, config)
            elif index == 0:
                manual_webots_message(args, "train")
                log_progress(
                    config,
                    "WEBOTS TRAIN manual_mode "
                    f"world={config['paths']['train_world']}",
                )
                if len(experiments) > 1:
                    print(
                        "Manual mode note: WheelchairEnv.close() stops robot clients. "
                        "If multiple experiments are selected, reload the training "
                        "world between experiments if clients do not reconnect."
                    )

            train_one_experiment(args, config, config_path, experiment)
        except WebotsLaunchError as exc:
            attempt = exc.attempt or attempt
            if attempt is not None and attempt not in config["webots"]["launch_attempts"]:
                config["webots"]["launch_attempts"].append(attempt)
            records = ensure_training_records(config, args)
            paths = experiment_paths(config["run_dir"], experiment)
            records[experiment]["status"] = "failed"
            records[experiment]["error_log"] = paths["error_log"]
            write_error_log(paths["error_log"], exc)
            save_training_metrics(records[experiment])
            append_config_event(config_path, config)
            print(f"Webots launch failed for {experiment}: {exc}", file=sys.stderr)
            log_progress(
                config,
                f"TRAIN ERROR experiment={experiment} error={exc}",
                experiment=experiment,
                level="ERROR",
            )
            if not args.continue_on_error:
                return 1
        except KeyboardInterrupt:
            records = ensure_training_records(config, args)
            records[experiment]["status"] = "interrupted"
            records[experiment]["interrupted"] = True
            append_config_event(config_path, config)
            print("Training interrupted by user.", file=sys.stderr)
            log_progress(
                config,
                f"TRAIN INTERRUPTED experiment={experiment}",
                experiment=experiment,
                level="ERROR",
            )
            return 130
        except Exception as exc:
            print(f"Training failed for {experiment}: {exc}", file=sys.stderr)
            log_progress(
                config,
                f"TRAIN ERROR experiment={experiment} error={exc}",
                experiment=experiment,
                level="ERROR",
            )
            if not args.continue_on_error:
                return 1
        finally:
            if process is not None and log_file is not None and attempt is not None:
                stop_webots(
                    process,
                    log_file,
                    attempt,
                    config=config,
                    experiment=experiment,
                )
                append_config_event(config_path, config)
                print(f"Stopped Webots training world; log: {attempt['log_path']}")

    return 0


def mean_value(values):
    return sum(values) / len(values) if values else 0.0


def median_value(values):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def std_value(values):
    if not values:
        return 0.0
    mean = mean_value(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def action_distribution(action_counts):
    total = sum(action_counts)
    if total == 0:
        return {label: 0.0 for label in ACTION_LABELS}
    return {
        label: 100 * action_counts[index] / total
        for index, label in enumerate(ACTION_LABELS)
    }


def make_evaluation_env(env_id, args):
    from stable_baselines3.common.monitor import Monitor
    from wheelchair_env import WheelchairEnv

    def _init():
        return Monitor(
            WheelchairEnv(
                env_id,
                lidar_dim=args.lidar_dim,
                reward_mode="default",
            )
        )

    return _init


def eval_robot_ids(config):
    robot_ids = config["validation"]["worlds"]["eval"].get("robot_ids", [])
    if robot_ids:
        return robot_ids
    return list(range(config["training"]["robots"]))


def evaluation_model_specs(args, config):
    specs = {
        "base_easy_model": {
            "model_path": with_zip_suffix(args.source_model),
            "model_base": args.source_model[:-4]
            if args.source_model.endswith(".zip")
            else args.source_model,
            "vecnormalize_path": args.source_vecnormalize,
            "type": "evaluation_only",
        }
    }
    for experiment in TRAINABLE_EXPERIMENTS:
        paths = experiment_paths(config["run_dir"], experiment)
        if args.eval_checkpoint == "best":
            model_path = paths["best_model"]
            model_base = paths["best_model_base"]
            vecnormalize_path = paths["best_vecnormalize"]
        else:
            model_path = paths["model"]
            model_base = paths["model_base"]
            vecnormalize_path = paths["vecnormalize"]
        specs[experiment] = {
            "model_path": model_path,
            "model_base": model_base,
            "vecnormalize_path": vecnormalize_path,
            "type": "trained",
            "checkpoint": args.eval_checkpoint,
        }
    return specs


def make_evaluation_record(config, experiment, args, spec):
    paths = experiment_paths(config["run_dir"], experiment)
    return {
        "status": "pending",
        "experiment": experiment,
        "type": spec["type"],
        "start_time": None,
        "end_time": None,
        "eval_world": args.eval_world,
        "eval_episodes_requested": args.eval_episodes,
        "eval_robot_ids": eval_robot_ids(config),
        "completed_episodes": 0,
        "completed_timesteps": 0,
        "wall_clock_seconds": 0.0,
        "model_path": spec["model_path"],
        "vecnormalize_path": spec["vecnormalize_path"],
        "progress_log": paths["progress_log"],
        "metrics": {},
        "outputs": {
            "metrics": paths["eval_metrics"],
            "episode_metrics": paths["eval_episode_metrics"],
            "action_distribution": paths["eval_action_distribution"],
            "trajectories": paths["eval_trajectories"],
        },
        "error_log": None,
        "interrupted": False,
    }


def ensure_evaluation_records(config, args):
    evaluation = config.setdefault("evaluation", {})
    records = evaluation.setdefault("experiments", {})
    for experiment, spec in evaluation_model_specs(args, config).items():
        records.setdefault(
            experiment,
            make_evaluation_record(config, experiment, args, spec),
        )
    return records


def archive_existing_flat_trajectory_logs(run_dir, experiment):
    existing = sorted(glob.glob(os.path.join("logs", "positions_*.csv")))
    if not existing:
        return []
    archive_dir = os.path.join(
        run_dir,
        experiment,
        "eval",
        "trajectories",
        f"pre_eval_archive_{timestamp()}",
    )
    os.makedirs(archive_dir, exist_ok=True)
    moved = []
    for path in existing:
        destination = os.path.join(archive_dir, os.path.basename(path))
        shutil.move(path, destination)
        moved.append(destination)
    return moved


def move_flat_trajectory_logs(trajectory_dir):
    os.makedirs(trajectory_dir, exist_ok=True)
    time.sleep(0.5)
    moved = []
    for path in sorted(glob.glob(os.path.join("logs", "positions_*.csv"))):
        destination = os.path.join(trajectory_dir, os.path.basename(path))
        shutil.move(path, destination)
        moved.append(destination)
    return moved


def write_episode_metrics_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "robot_id",
                "episode_id",
                "success",
                "collision",
                "timeout",
                "goal_reached",
                "episode_steps",
                "episode_reward",
                "final_step_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_action_distribution_csv(path, action_counts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    distribution = action_distribution(action_counts)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["action", "count", "percent"])
        for index, label in enumerate(ACTION_LABELS):
            writer.writerow([label, action_counts[index], f"{distribution[label]:.2f}"])


def aggregate_evaluation_metrics(rows, action_counts, completed_timesteps):
    rewards = [float(row["episode_reward"]) for row in rows]
    successful_rows = [row for row in rows if row["success"]]
    time_to_goal = [float(row["episode_steps"]) for row in successful_rows]
    episode_lengths = [float(row["episode_steps"]) for row in rows]
    total_episodes = len(rows)
    success_count = sum(1 for row in rows if row["success"])
    collision_count = sum(1 for row in rows if row["collision"])
    timeout_count = sum(1 for row in rows if row["timeout"])
    distribution = action_distribution(action_counts)
    metrics = {
        "reward_mean": mean_value(rewards),
        "reward_median": median_value(rewards),
        "reward_std": std_value(rewards),
        "success_rate": 100 * success_count / total_episodes if total_episodes else 0.0,
        "collision_rate": (
            100 * collision_count / total_episodes if total_episodes else 0.0
        ),
        "timeout_rate": 100 * timeout_count / total_episodes if total_episodes else 0.0,
        "avg_time_to_goal": mean_value(time_to_goal),
        "median_time_to_goal": median_value(time_to_goal),
        "avg_episode_steps": mean_value(episode_lengths),
        "completed_episodes": total_episodes,
        "completed_timesteps": completed_timesteps,
        "action_counts": {
            label: action_counts[index] for index, label in enumerate(ACTION_LABELS)
        },
        "action_distribution_percent": distribution,
    }
    return metrics


def timeout_rows_for_partial_episodes(
    experiment,
    robot_ids,
    episode_counts,
    episode_steps,
    episode_rewards,
):
    rows = []
    for index, steps in enumerate(episode_steps):
        if steps <= 0:
            continue
        rows.append({
            "experiment": experiment,
            "robot_id": robot_ids[index],
            "episode_id": episode_counts[index] + 1,
            "success": False,
            "collision": False,
            "timeout": True,
            "goal_reached": False,
            "episode_steps": int(steps),
            "episode_reward": float(episode_rewards[index]),
            "final_step_reward": 0.0,
        })
    return rows


def evaluate_one_experiment(args, config, config_path, experiment):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    patch_numpy_pickle_modules()
    specs = evaluation_model_specs(args, config)
    spec = specs[experiment]
    paths = experiment_paths(config["run_dir"], experiment)
    records = ensure_evaluation_records(config, args)
    record = records[experiment]
    robot_ids = eval_robot_ids(config)
    env = None
    start_monotonic = time.monotonic()
    episode_rows = []
    action_counts = [0] * len(ACTION_LABELS)
    completed_steps = 0
    safety_step_cap = max(5000, args.eval_episodes * 20000)
    eval_progress_episode_interval = max(1, args.eval_episodes // 10)
    last_progress_episode_count = 0
    last_progress_step_count = 0
    record.update({
        "status": "running",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "end_time": None,
        "completed_episodes": 0,
        "completed_timesteps": 0,
        "wall_clock_seconds": 0.0,
        "error_log": None,
        "interrupted": False,
        "eval_robot_ids": robot_ids,
    })
    append_config_event(config_path, config)
    log_progress(
        config,
        "EXPERIMENT START "
        f"experiment={experiment} phase=eval checkpoint={args.eval_checkpoint}",
        experiment=experiment,
    )
    log_progress(
        config,
        "EVAL START "
        f"experiment={experiment} episodes={args.eval_episodes} "
        f"world={config['paths']['eval_world']} "
        f"model={spec['model_path']} vecnormalize={spec['vecnormalize_path']}",
        experiment=experiment,
    )

    try:
        if not os.path.exists(spec["model_path"]):
            raise FileNotFoundError(f"model does not exist: {spec['model_path']}")
        if not os.path.exists(spec["vecnormalize_path"]):
            raise FileNotFoundError(
                f"VecNormalize file does not exist: {spec['vecnormalize_path']}"
            )

        archive_existing_flat_trajectory_logs(config["run_dir"], experiment)
        env = DummyVecEnv([make_evaluation_env(robot_id, args) for robot_id in robot_ids])
        env = VecNormalize.load(spec["vecnormalize_path"], env)
        env.training = False
        env.norm_reward = False

        model = PPO.load(spec["model_base"], env=env)
        obs = env.reset()
        episode_counts = [0] * len(robot_ids)
        episode_rewards = [0.0] * len(robot_ids)
        episode_steps = [0] * len(robot_ids)

        while len(episode_rows) < args.eval_episodes and completed_steps < safety_step_cap:
            action, _ = model.predict(obs, deterministic=True)
            for action_id in action:
                action_counts[int(action_id)] += 1

            obs, rewards, dones, infos = env.step(action)
            completed_steps += 1

            for index in range(len(robot_ids)):
                episode_rewards[index] += float(rewards[index])
                episode_steps[index] += 1

            for index, done in enumerate(dones):
                if not done:
                    continue
                info = infos[index]
                episode_counts[index] += 1
                success = bool(info.get("is_success", False))
                collision = bool(info.get("collision", False))
                goal_reached = bool(info.get("goal_reached", False))
                final_step_reward = float(info.get("reward", 0.0))
                episode_rows.append({
                    "experiment": experiment,
                    "robot_id": robot_ids[index],
                    "episode_id": episode_counts[index],
                    "success": success,
                    "collision": collision,
                    "timeout": False,
                    "goal_reached": goal_reached,
                    "episode_steps": int(info.get("time_step", episode_steps[index])),
                    "episode_reward": float(episode_rewards[index]),
                    "final_step_reward": final_step_reward,
                })
                episode_rewards[index] = 0.0
                episode_steps[index] = 0
                if len(episode_rows) >= args.eval_episodes:
                    break

            if len(episode_rows) != last_progress_episode_count and (
                len(episode_rows) % eval_progress_episode_interval == 0
                or completed_steps - last_progress_step_count
                >= max(1, args.progress_log_freq)
                or len(episode_rows) >= args.eval_episodes
            ):
                log_progress(
                    config,
                    "EVAL PROGRESS "
                    f"experiment={experiment} "
                    f"completed_episodes={len(episode_rows)}/{args.eval_episodes} "
                    f"completed_steps={completed_steps}",
                    experiment=experiment,
                )
                last_progress_episode_count = len(episode_rows)
                last_progress_step_count = completed_steps

        if len(episode_rows) < args.eval_episodes:
            episode_rows.extend(
                timeout_rows_for_partial_episodes(
                    experiment,
                    robot_ids,
                    episode_counts,
                    episode_steps,
                    episode_rewards,
                )
            )

        final_episode_rows = episode_rows[: args.eval_episodes]
        metrics = aggregate_evaluation_metrics(
            final_episode_rows,
            action_counts,
            completed_steps,
        )
        moved_trajectories = move_flat_trajectory_logs(paths["eval_trajectories"])
        metrics["moved_trajectories"] = moved_trajectories
        record["status"] = "completed"
        record["metrics"] = metrics
        record["completed_episodes"] = metrics["completed_episodes"]
        record["completed_timesteps"] = completed_steps
        write_episode_metrics_csv(paths["eval_episode_metrics"], final_episode_rows)
        write_action_distribution_csv(paths["eval_action_distribution"], action_counts)
        write_json(paths["eval_metrics"], record)
        log_progress(
            config,
            "EVAL END "
            f"experiment={experiment} "
            f"completed_episodes={metrics['completed_episodes']} "
            f"completed_timesteps={completed_steps} "
            f"success_rate={safe_metric_value(metrics.get('success_rate'))} "
            f"collision_rate={safe_metric_value(metrics.get('collision_rate'))} "
            f"timeout_rate={safe_metric_value(metrics.get('timeout_rate'))} "
            f"avg_time_to_goal={safe_metric_value(metrics.get('avg_time_to_goal'))} "
            f"metrics={paths['eval_metrics']}",
            experiment=experiment,
        )
        log_progress(
            config,
            f"EXPERIMENT END experiment={experiment} phase=eval status=completed",
            experiment=experiment,
        )
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["interrupted"] = True
        log_progress(
            config,
            f"EVAL INTERRUPTED experiment={experiment}",
            experiment=experiment,
            level="ERROR",
        )
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error_log"] = paths["eval_error_log"]
        write_error_log(paths["eval_error_log"], exc)
        log_progress(
            config,
            "EVAL ERROR "
            f"experiment={experiment} error={exc} error_log={paths['eval_error_log']}",
            experiment=experiment,
            level="ERROR",
        )
        raise
    finally:
        record["end_time"] = datetime.now().isoformat(timespec="seconds")
        record["wall_clock_seconds"] = round(time.monotonic() - start_monotonic, 3)
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                record.setdefault("warnings", []).append(
                    f"env.close() failed: {close_exc}"
                )
        write_json(paths["eval_metrics"], record)
        append_config_event(config_path, config)

    print(f"Evaluation completed for {experiment}: {paths['eval_metrics']}")


def selected_evaluation_experiments(args, config):
    if args.skip_eval:
        return []
    specs = evaluation_model_specs(args, config)
    records = ensure_evaluation_records(config, args)
    selected = []
    for experiment in ALL_EXPERIMENTS:
        if experiment not in specs:
            continue
        spec = specs[experiment]
        missing = [
            path
            for path in [spec["model_path"], spec["vecnormalize_path"]]
            if not os.path.exists(path)
        ]
        if missing:
            if args.eval_checkpoint == "best" and experiment in TRAINABLE_EXPERIMENTS:
                raise FileNotFoundError(
                    f"--eval-checkpoint best requested for {experiment}, but missing: "
                    + ", ".join(missing)
                )
            record = records[experiment]
            record["status"] = "skipped_missing_artifacts"
            record["missing_artifacts"] = missing
            write_json(record["outputs"]["metrics"], record)
            print(
                f"Skipping evaluation for {experiment}; missing artifacts: "
                + ", ".join(missing)
            )
            log_progress(
                config,
                "EVAL SKIP "
                f"experiment={experiment} reason=missing_artifacts "
                f"missing={','.join(missing)}",
                experiment=experiment,
            )
            continue
        selected.append(experiment)
    return selected


def run_evaluation_experiments(args, config, config_path):
    ensure_evaluation_records(config, args)
    experiments = selected_evaluation_experiments(args, config)
    if not experiments:
        print("Evaluation skipped by --skip-eval.")
        log_progress(config, "EVAL SKIP reason=skip_eval_or_no_available_artifacts")
        append_config_event(config_path, config)
        return 0

    for index, experiment in enumerate(experiments):
        process = None
        log_file = None
        attempt = None
        try:
            if args.launch_webots:
                label = "eval" if len(experiments) == 1 else f"eval_{experiment}"
                print(f"Launching Webots evaluation world for {experiment}.")
                process, log_file, attempt = launch_webots(
                    args,
                    config["run_dir"],
                    label,
                    config["paths"]["eval_world"],
                    config=config,
                    experiment=experiment,
                )
                config["webots"]["launch_attempts"].append(attempt)
                append_config_event(config_path, config)
            elif index == 0:
                manual_webots_message(args, "eval")
                log_progress(
                    config,
                    "WEBOTS EVAL manual_mode "
                    f"world={config['paths']['eval_world']}",
                )
                if len(experiments) > 1:
                    print(
                        "Manual mode note: WheelchairEnv.close() stops robot clients. "
                        "If multiple evaluations are selected, reload the evaluation "
                        "world between models if clients do not reconnect."
                    )

            evaluate_one_experiment(args, config, config_path, experiment)
        except WebotsLaunchError as exc:
            attempt = exc.attempt or attempt
            if attempt is not None and attempt not in config["webots"]["launch_attempts"]:
                config["webots"]["launch_attempts"].append(attempt)
            records = ensure_evaluation_records(config, args)
            paths = experiment_paths(config["run_dir"], experiment)
            records[experiment]["status"] = "failed"
            records[experiment]["error_log"] = paths["eval_error_log"]
            write_error_log(paths["eval_error_log"], exc)
            write_json(paths["eval_metrics"], records[experiment])
            append_config_event(config_path, config)
            print(f"Webots launch failed for {experiment}: {exc}", file=sys.stderr)
            log_progress(
                config,
                f"EVAL ERROR experiment={experiment} error={exc}",
                experiment=experiment,
                level="ERROR",
            )
            if not args.continue_on_error:
                return 1
        except KeyboardInterrupt:
            records = ensure_evaluation_records(config, args)
            records[experiment]["status"] = "interrupted"
            records[experiment]["interrupted"] = True
            append_config_event(config_path, config)
            print("Evaluation interrupted by user.", file=sys.stderr)
            log_progress(
                config,
                f"EVAL INTERRUPTED experiment={experiment}",
                experiment=experiment,
                level="ERROR",
            )
            return 130
        except Exception as exc:
            print(f"Evaluation failed for {experiment}: {exc}", file=sys.stderr)
            log_progress(
                config,
                f"EVAL ERROR experiment={experiment} error={exc}",
                experiment=experiment,
                level="ERROR",
            )
            if not args.continue_on_error:
                return 1
        finally:
            if process is not None and log_file is not None and attempt is not None:
                stop_webots(
                    process,
                    log_file,
                    attempt,
                    config=config,
                    experiment=experiment,
                )
                append_config_event(config_path, config)
                print(f"Stopped Webots evaluation world; log: {attempt['log_path']}")

    return 0


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def csv_value(value):
    if value is None:
        return ""
    return value


def markdown_value(value):
    if value is None:
        return "N/A"
    return str(value)


def experiment_role(experiment):
    roles = {
        "base_easy_model": "eval-only, trained previously on easy map",
        "scratch": "trained from zero on transfer-finetune map",
        "transfer_finetune": (
            "initialized from base model, fine-tuned with transfer_finetune reward"
        ),
        "transfer_goal_direction": (
            "initialized from base model, fine-tuned with transfer_goal_direction reward"
        ),
    }
    return roles[experiment]


def trajectory_points(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        sort_key = next(
            (name for name in ["timestamp", "time", "step", "t"] if name in fieldnames),
            None,
        )
        for index, row in enumerate(reader):
            try:
                x = float(row.get("x"))
                y = float(row.get("y"))
            except (TypeError, ValueError):
                continue
            order = index
            if sort_key is not None:
                try:
                    order = float(row.get(sort_key))
                except (TypeError, ValueError):
                    order = index
            rows.append((order, x, y))
    rows.sort(key=lambda item: item[0])
    points = []
    for _, x, y in rows:
        point = (x, y)
        if points and points[-1] == point:
            continue
        points.append(point)
    return points


def trajectory_metrics_for_experiment(run_dir, experiment, warnings):
    trajectory_dir = experiment_paths(run_dir, experiment)["eval_trajectories"]
    paths = sorted(glob.glob(os.path.join(trajectory_dir, "*.csv")))
    metrics = []
    for path in paths:
        try:
            points = trajectory_points(path)
            if len(points) < 2:
                continue
            path_length = 0.0
            direction_changes = 0
            previous_direction = None
            for left, right in zip(points, points[1:]):
                dx = right[0] - left[0]
                dy = right[1] - left[1]
                segment = (dx * dx + dy * dy) ** 0.5
                if segment <= 0:
                    continue
                path_length += segment
                direction = (round(dx / segment, 3), round(dy / segment, 3))
                if previous_direction is not None and direction != previous_direction:
                    direction_changes += 1
                previous_direction = direction
            displacement = (
                (points[-1][0] - points[0][0]) ** 2
                + (points[-1][1] - points[0][1]) ** 2
            ) ** 0.5
            efficiency = displacement / path_length if path_length > 0 else None
            if efficiency is not None:
                efficiency = max(0.0, min(1.0, efficiency))
            metrics.append({
                "path_length": path_length,
                "straight_line_displacement": displacement,
                "path_efficiency": efficiency,
                "direction_changes": direction_changes,
                "turning_ratio": direction_changes / max(len(points) - 1, 1),
            })
        except Exception as exc:
            warnings.append(f"Could not parse trajectory {path}: {exc}")
    if not metrics:
        return {
            "avg_path_length": None,
            "avg_straight_line_displacement": None,
            "avg_path_efficiency": None,
            "avg_direction_changes": None,
            "avg_turning_ratio": None,
        }
    return {
        "avg_path_length": mean_value([m["path_length"] for m in metrics]),
        "avg_straight_line_displacement": mean_value(
            [m["straight_line_displacement"] for m in metrics]
        ),
        "avg_path_efficiency": mean_value(
            [m["path_efficiency"] for m in metrics if m["path_efficiency"] is not None]
        ),
        "avg_direction_changes": mean_value([m["direction_changes"] for m in metrics]),
        "avg_turning_ratio": mean_value([m["turning_ratio"] for m in metrics]),
    }


def summary_row_for_experiment(args, config, experiment, warnings):
    paths = experiment_paths(config["run_dir"], experiment)
    training = read_json(paths["training_metrics"]) or {}
    evaluation = read_json(paths["eval_metrics"]) or {}
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    trajectory_metrics = trajectory_metrics_for_experiment(
        config["run_dir"], experiment, warnings
    )
    if not any(value is not None for value in trajectory_metrics.values()):
        warnings.append(f"No trajectory metrics available for {experiment}.")
    source_model = config["paths"].get("source_model") or with_zip_suffix(args.source_model)
    return {
        "experiment": experiment,
        "role": experiment_role(experiment),
        "trained_in_pipeline": experiment in TRAINABLE_EXPERIMENTS,
        "source_model": source_model if experiment != "scratch" else None,
        "train_world": (
            config["paths"].get("train_world")
            if experiment in TRAINABLE_EXPERIMENTS
            else None
        ),
        "eval_world": config["paths"].get("eval_world"),
        "training_reward_mode": (
            reward_mode_for_experiment(experiment)
            if experiment in TRAINABLE_EXPERIMENTS
            else None
        ),
        "evaluation_reward_mode": "default",
        "training_status": training.get("status"),
        "evaluation_status": evaluation.get("status"),
        "training_timesteps": training.get("actual_timesteps"),
        "wall_clock_train_time": training.get("wall_clock_seconds"),
        "early_stopped": (training.get("early_stopping") or {}).get("early_stopped"),
        "actual_stop_timestep": (
            training.get("early_stopping") or {}
        ).get("actual_stop_timestep"),
        "stop_reason": (training.get("early_stopping") or {}).get("stop_reason"),
        "default_eval_reward_mean": metrics.get("reward_mean"),
        "default_eval_reward_std": metrics.get("reward_std"),
        "default_eval_reward_median": metrics.get("reward_median"),
        "success_rate": metrics.get("success_rate"),
        "collision_rate": metrics.get("collision_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "avg_time_to_goal": metrics.get("avg_time_to_goal"),
        "median_time_to_goal": metrics.get("median_time_to_goal"),
        "avg_episode_steps": metrics.get("avg_episode_steps"),
        "completed_eval_episodes": metrics.get("completed_episodes"),
        "completed_eval_timesteps": metrics.get("completed_timesteps"),
        "forward_action_pct": (metrics.get("action_distribution_percent") or {}).get(
            "forward"
        ),
        "turn_action_pct": sum(
            (metrics.get("action_distribution_percent") or {}).get(action, 0.0)
            for action in ["forward_left", "forward_right", "left", "right"]
        )
        if metrics.get("action_distribution_percent")
        else None,
        "stop_action_pct": (metrics.get("action_distribution_percent") or {}).get(
            "stop"
        ),
        **trajectory_metrics,
    }


SUMMARY_FIELDS = [
    "experiment",
    "role",
    "trained_in_pipeline",
    "source_model",
    "train_world",
    "eval_world",
    "training_reward_mode",
    "evaluation_reward_mode",
    "training_status",
    "evaluation_status",
    "training_timesteps",
    "wall_clock_train_time",
    "early_stopped",
    "actual_stop_timestep",
    "stop_reason",
    "default_eval_reward_mean",
    "default_eval_reward_std",
    "default_eval_reward_median",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_time_to_goal",
    "median_time_to_goal",
    "avg_episode_steps",
    "completed_eval_episodes",
    "completed_eval_timesteps",
    "forward_action_pct",
    "turn_action_pct",
    "stop_action_pct",
    "avg_path_length",
    "avg_straight_line_displacement",
    "avg_path_efficiency",
    "avg_direction_changes",
    "avg_turning_ratio",
]


def write_summary_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in SUMMARY_FIELDS})


def safe_plot(warnings, plot_func, *args):
    try:
        return plot_func(*args)
    except Exception as exc:
        warnings.append(f"Plot skipped: {exc}")
        return None


def plot_bar(rows, key, title, ylabel, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [row for row in rows if row.get(key) is not None]
    if not plot_rows:
        raise ValueError(f"no values for {key}")
    labels = [row["experiment"] for row in plot_rows]
    values = [float(row[key]) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#3267a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_action_distribution_comparison(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_keys = ["forward_action_pct", "turn_action_pct", "stop_action_pct"]
    labels = [row["experiment"] for row in rows]
    x_positions = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, key in enumerate(action_keys):
        values = [float(row.get(key) or 0.0) for row in rows]
        shifted = [pos + (offset - 1) * width for pos in x_positions]
        ax.bar(shifted, values, width=width, label=key)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Action share (%)")
    ax.set_title("Action Distribution Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def generate_report_markdown(config, rows, warnings, plot_paths):
    lines = [
        "# Transfer Comparison Report",
        "",
        "Training reward is not directly comparable across all models because some models were trained with shaped rewards. Final conclusions should prioritize evaluation metrics collected under the same evaluation setup.",
        "",
        "Early stopping evaluation, when enabled, is only for stopping and checkpointing on the active training world. The official final comparison is the Stage 4 evaluation on the real-world test map.",
        "",
        "## Experiment Setup",
        "",
        f"- Training world: `{config['paths']['train_world']}`",
        f"- Evaluation world: `{config['paths']['eval_world']}`",
        f"- Source model: `{config['paths']['source_model']}`",
        "",
        "## Summary",
        "",
    ]
    columns = [
        "experiment",
        "role",
        "default_eval_reward_mean",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "avg_time_to_goal",
        "avg_episode_steps",
        "forward_action_pct",
        "turn_action_pct",
        "stop_action_pct",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_value(row.get(column)) for column in columns)
            + " |"
        )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## Plots", ""])
    if plot_paths:
        for path in plot_paths:
            rel_path = os.path.relpath(path, config["run_dir"]).replace(os.sep, "/")
            lines.append(f"- `{rel_path}`")
    else:
        lines.append("- No plots generated.")
    return "\n".join(lines) + "\n"


def generate_report_outputs(args, config, config_path):
    log_progress(config, "REPORT START")
    warnings = ["TensorBoard curves were not parsed for this run."]
    rows = [
        summary_row_for_experiment(args, config, experiment, warnings)
        for experiment in ALL_EXPERIMENTS
    ]
    summary_json_path = os.path.join(config["run_dir"], "summary.json")
    summary_csv_path = os.path.join(config["run_dir"], "summary.csv")
    write_json(summary_json_path, {"rows": rows, "warnings": warnings})
    write_summary_csv(summary_csv_path, rows)

    plots_dir = os.path.join(config["run_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_paths = []
    plot_specs = [
        (
            "default_eval_reward_mean",
            "Default Evaluation Reward Mean",
            "Reward",
            "default_eval_reward_mean.png",
        ),
        ("success_rate", "Success Rate", "Success (%)", "success_rate.png"),
        ("collision_rate", "Collision Rate", "Collision (%)", "collision_rate.png"),
        (
            "avg_time_to_goal",
            "Average Time to Goal",
            "Steps",
            "avg_time_to_goal.png",
        ),
        (
            "avg_episode_steps",
            "Average Episode Length",
            "Steps",
            "avg_episode_steps.png",
        ),
    ]
    for key, title, ylabel, filename in plot_specs:
        path = safe_plot(
            warnings,
            plot_bar,
            rows,
            key,
            title,
            ylabel,
            os.path.join(plots_dir, filename),
        )
        if path:
            plot_paths.append(path)
    action_plot = safe_plot(
        warnings,
        plot_action_distribution_comparison,
        rows,
        os.path.join(plots_dir, "action_distribution_comparison.png"),
    )
    if action_plot:
        plot_paths.append(action_plot)

    report_path = os.path.join(config["run_dir"], "reports", "comparison_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(generate_report_markdown(config, rows, warnings, plot_paths))

    config["reporting"] = {
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
        "comparison_report": report_path,
        "plots": plot_paths,
        "warnings": warnings,
    }
    append_config_event(config_path, config)
    for warning in warnings:
        log_progress(config, f"REPORT WARNING {warning}")
    log_progress(
        config,
        "REPORT END "
        f"summary_csv={summary_csv_path} summary_json={summary_json_path} "
        f"report={report_path}",
    )
    print(f"Wrote summary CSV to {summary_csv_path}")
    print(f"Wrote summary JSON to {summary_json_path}")
    print(f"Wrote comparison report to {report_path}")
    return 0


def build_config(args, phase, run_dir, resumed, validations):
    return {
        "schema_stage": 6,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        "phase": phase,
        "check_only": phase == "check",
        "resume_run": args.resume_run,
        "resumed": resumed,
        "run_dir": run_dir,
        "paths": {
            "train_world": args.train_world,
            "eval_world": args.eval_world,
            "output_root": args.output_root,
            "source_model": with_zip_suffix(args.source_model),
            "source_vecnormalize": args.source_vecnormalize,
        },
        "experiments": experiment_matrix(args),
        "training": {
            "max_timesteps": args.max_timesteps,
            "robots": args.robots,
            "lidar_dim": args.lidar_dim,
            "learning_rate_scratch": args.learning_rate_scratch,
            "learning_rate_transfer": args.learning_rate_transfer,
            "seed": args.seed,
            "skip_training": args.skip_training,
        },
        "evaluation": {
            "eval_episodes": args.eval_episodes,
            "skip_eval": args.skip_eval,
            "eval_checkpoint": args.eval_checkpoint,
            "deterministic_actions": True,
            "vecnormalize_training": False,
            "update_normalization_stats": False,
        },
        "progress_logging": {
            "progress_log_freq": args.progress_log_freq,
            "root_progress_log": os.path.join(run_dir, "logs", "progress.log"),
            "per_experiment_progress_logs": {
                experiment: experiment_paths(run_dir, experiment)["progress_log"]
                for experiment in ALL_EXPERIMENTS
            },
        },
        "webots": {
            "launch_webots": args.launch_webots,
            "webots_bin": args.webots_bin,
            "webots_mode": args.webots_mode,
            "webots_no_rendering": args.webots_no_rendering,
            "webots_startup_wait": args.webots_startup_wait,
            "planned_commands": planned_webots_commands(args),
            "launch_attempts": [],
            "stage6_note": (
                "Stage 6 may launch Webots for training and evaluation, generate "
                "reports, and optionally use early stopping."
            ),
        },
        "early_stopping": {
            "enabled": args.early_stop,
            "metric": args.early_stop_metric,
            "eval_freq": args.eval_freq,
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "min_timesteps": args.early_stop_min_timesteps,
            "eval_episodes": args.early_stop_eval_episodes,
            "metric_direction": early_stop_metric_direction(args.early_stop_metric),
            "stage6_note": (
                "Early stopping uses the active training world only; final official "
                "comparison still uses the separate evaluation world."
            ),
        },
        "failure_recovery": {
            "continue_on_error": args.continue_on_error,
            "default": "fail_fast",
        },
        "validation": validations,
        "environment": package_versions(),
    }


def print_check_summary(config):
    print("PPO transfer comparison pipeline check")
    print(f"  Run directory: {config['run_dir']}")
    print(f"  Phase: {config['phase']}")
    print(f"  Training world: {config['paths']['train_world']}")
    print(f"  Evaluation world: {config['paths']['eval_world']}")
    print(f"  Source model: {config['paths']['source_model']}")
    print(f"  Source VecNormalize: {config['paths']['source_vecnormalize']}")
    print(f"  Robots requested: {config['training']['robots']}")
    print(f"  LiDAR dim: {config['training']['lidar_dim']}")
    print("  Experiments:")
    for name, experiment in config["experiments"].items():
        if name == "base_easy_model":
            print("    - base_easy_model: evaluation-only baseline")
        else:
            selected = "selected" if experiment["selected_for_training"] else "skipped"
            print(
                f"    - {name}: {selected}, reward_mode={experiment['reward_mode']}"
            )

    train_ids = config["validation"]["worlds"]["train"]["robot_ids"]
    eval_ids = config["validation"]["worlds"]["eval"]["robot_ids"]
    print(f"  Training world robot IDs: {train_ids}")
    print(f"  Evaluation world robot IDs: {eval_ids}")

    if config["validation"]["warnings"]:
        print("  Warnings:")
        for warning in config["validation"]["warnings"]:
            print(f"    - {warning}")

    print(f"  Wrote config: {os.path.join(config['run_dir'], 'config.json')}")


def run_stage6(args):
    phase = resolve_phase(args)
    run_dir, resumed = make_run_dir(args, phase)
    create_pipeline_folders(run_dir)
    validations = validate_paths(args)
    config = build_config(args, phase, run_dir, resumed, validations)
    config_path = os.path.join(run_dir, "config.json")
    if resumed and os.path.exists(config_path):
        existing_config = read_json(config_path) or {}
        existing_config.update({
            "command": sys.argv,
            "phase": phase,
            "check_only": phase == "check",
            "resume_run": args.resume_run,
            "resumed": resumed,
            "run_dir": run_dir,
            "validation": validations,
        })
        existing_config.setdefault("paths", config["paths"])
        existing_config.setdefault("training", config["training"])
        existing_config.setdefault("evaluation", config["evaluation"])
        existing_config["progress_logging"] = config["progress_logging"]
        existing_config.setdefault("webots", config["webots"])
        existing_config.setdefault("early_stopping", config["early_stopping"])
        existing_config.setdefault("failure_recovery", config["failure_recovery"])
        existing_config.setdefault("environment", config["environment"])
        config = existing_config
    write_json(config_path, config)
    log_progress(
        config,
        f"PIPELINE START run={run_dir} phase={phase}",
    )
    log_progress(
        config,
        "PIPELINE CONFIG "
        f"train_world={config['paths']['train_world']} "
        f"eval_world={config['paths']['eval_world']} "
        f"experiments={','.join(args.experiments)} "
        f"webots={'automatic' if args.launch_webots else 'manual'} "
        f"progress_log_freq={args.progress_log_freq}",
    )

    if validations["errors"]:
        print_check_summary(config)
        for error in validations["errors"]:
            print(f"Error: {error}", file=sys.stderr)
            log_progress(config, f"VALIDATION ERROR {error}", level="ERROR")
        log_progress(config, "PIPELINE END status=failed")
        return 1

    if phase == "check":
        print_check_summary(config)
        print("Check mode complete. No Webots, training, or evaluation was started.")
        log_progress(config, "PIPELINE END status=completed")
        return 0

    print_check_summary(config)
    if phase in ("train", "all"):
        training_result = run_training_experiments(args, config, config_path)
        if training_result != 0:
            status = "interrupted" if training_result == 130 else "failed"
            log_progress(config, f"PIPELINE END status={status}")
            return training_result
        if phase == "train":
            print(
                "Training phase complete. Reporting, plots, TensorBoard parsing, "
                "and final comparison reporting can be generated with --phase report."
            )
            log_progress(config, "PIPELINE END status=completed")
            return 0

    if phase in ("eval", "all"):
        evaluation_result = run_evaluation_experiments(args, config, config_path)
        if evaluation_result != 0:
            status = "interrupted" if evaluation_result == 130 else "failed"
            log_progress(config, f"PIPELINE END status={status}")
            return evaluation_result
        if phase == "eval":
            print("Evaluation phase complete. Use --phase report to generate summaries.")
            log_progress(config, "PIPELINE END status=completed")
            return 0

    if phase in ("report", "all"):
        report_result = generate_report_outputs(args, config, config_path)
        status = "completed" if report_result == 0 else "failed"
        log_progress(config, f"PIPELINE END status={status}")
        return report_result

    print(
        f"Phase '{phase}' is not implemented beyond Stage 6 scaffolding. "
        "No training, evaluation, reporting, or plotting was started."
    )
    log_progress(config, "PIPELINE END status=completed")
    return 0


def main():
    args = make_parser().parse_args()
    try:
        return run_stage6(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
