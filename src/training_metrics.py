import csv
import os
import time
from collections import Counter
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        nn_type,
        lidar_dim,
        n_actions=6,
        best_model_path=None,
        best_vecnorm_path=None,
        verbose=1,
    ):
        super().__init__(verbose)
        self.nn_type = nn_type
        self.lidar_dim = lidar_dim
        self.n_actions = n_actions
        self.best_model_path = best_model_path
        self.best_vecnorm_path = best_vecnorm_path
        self.best_success_rate = -np.inf
        self.best_collision_rate = np.inf
        self.best_reward_mean = -np.inf
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{nn_type}_lidar{lidar_dim}_{self.run_id}"
        self.csv_path = f"logs/training_metrics_{self.run_name}.csv"
        self.episodes_path = f"logs/training_episodes_{self.run_name}.csv"
        self.figures_dir = f"figures/training/{nn_type}/{self.run_id}"
        self.rows = []
        self.episodes = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.actions = Counter()

    def _on_training_start(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        print(f"[{self.nn_type}] metrics run: {self.run_id}")
        print(f"[{self.nn_type}] figures: {self.figures_dir}")
        print(f"[{self.nn_type}] metrics csv: {self.csv_path}")
        n_envs = self.training_env.num_envs
        self.ep_rewards = [0.0] * n_envs
        self.ep_lengths = [0] * n_envs

    def _on_rollout_start(self):
        self.started_at = time.time()
        self.rollout_episodes = []
        self.actions = Counter()

    def _on_step(self):
        actions = np.asarray(self.locals.get("actions", [])).reshape(-1)
        rewards = np.asarray(self.locals.get("rewards", [])).reshape(-1)
        dones = np.asarray(self.locals.get("dones", [])).reshape(-1)
        infos = self.locals.get("infos", [])

        self.actions.update(int(a) for a in actions)
        for i, reward in enumerate(rewards):
            self.ep_rewards[i] += float(reward)
            self.ep_lengths[i] += 1

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            row = {
                "timestep": self.num_timesteps,
                "env_id": int(info.get("env_id", i)),
                "reward": self.ep_rewards[i],
                "length": int(info.get("time_step", self.ep_lengths[i])),
                "success": int(bool(info.get("is_success", False))),
                "collision": int(bool(info.get("collision", False))),
                "goal": int(bool(info.get("goal_reached", False))),
                "timeout": int(bool(info.get("time_limit_reached", False))),
            }
            self.episodes.append(row)
            self.rollout_episodes.append(row)
            self.ep_rewards[i] = 0.0
            self.ep_lengths[i] = 0
        return True

    def _on_rollout_end(self):
        episodes = len(self.rollout_episodes)
        action_total = sum(self.actions.values())
        rewards = [e["reward"] for e in self.rollout_episodes]
        lengths = [e["length"] for e in self.rollout_episodes]
        successes = sum(e["success"] for e in self.rollout_episodes)
        collisions = sum(e["collision"] for e in self.rollout_episodes)
        goals = sum(e["goal"] for e in self.rollout_episodes)
        timeouts = sum(e["timeout"] for e in self.rollout_episodes)

        row = {
            "timestep": self.num_timesteps,
            "episodes": episodes,
            "reward_mean": np.mean(rewards) if rewards else np.nan,
            "length_mean": np.mean(lengths) if lengths else np.nan,
            "success_rate": successes / episodes if episodes else np.nan,
            "collision_rate": collisions / episodes if episodes else np.nan,
            "goals": goals,
            "collisions": collisions,
            "timeouts": timeouts,
            "steps_per_second": action_total / max(time.time() - self.started_at, 1e-9),
        }
        for a in range(self.n_actions):
            row[f"action_{a}_rate"] = self.actions[a] / action_total if action_total else 0.0
        self.rows.append(row)

        for key, value in row.items():
            if key != "timestep" and not pd.isna(value):
                self.logger.record(f"custom/{self.nn_type}/{key}", value)
        if self.best_success_rate > -np.inf:
            self.logger.record(
                f"custom/{self.nn_type}/best_success_rate",
                self.best_success_rate,
            )

        print(
            f"[{self.nn_type}] t={row['timestep']} reward={row['reward_mean']:.2f} "
            f"success={100 * row['success_rate']:.1f}% collisions={collisions} "
            f"goals={goals} timeouts={timeouts}",
            flush=True,
        )
        self._save_best_checkpoint(row)

    def _on_training_end(self):
        self.save()

    def save(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        if self.rows:
            self._write_csv(self.csv_path, self.rows)
            self._plot_all(pd.DataFrame(self.rows))
        if self.episodes:
            self._write_csv(self.episodes_path, self.episodes)

    def _save_best_checkpoint(self, row):
        success_rate = row["success_rate"]
        if pd.isna(success_rate):
            return

        collision_rate = row["collision_rate"]
        reward_mean = row["reward_mean"]
        collision_rate = float(collision_rate) if not pd.isna(collision_rate) else np.inf
        reward_mean = float(reward_mean) if not pd.isna(reward_mean) else -np.inf
        success_rate = float(success_rate)

        better_success = success_rate > self.best_success_rate
        tied_success = np.isclose(success_rate, self.best_success_rate)
        better_collision = collision_rate < self.best_collision_rate
        tied_collision = np.isclose(collision_rate, self.best_collision_rate)
        better_reward = reward_mean > self.best_reward_mean

        if not (
            better_success
            or (tied_success and better_collision)
            or (tied_success and tied_collision and better_reward)
        ):
            return

        self.best_success_rate = float(success_rate)
        self.best_collision_rate = collision_rate
        self.best_reward_mean = reward_mean
        if self.best_model_path is None:
            return

        model_dir = os.path.dirname(self.best_model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        self.model.save(self.best_model_path)

        if self.best_vecnorm_path is not None and hasattr(self.training_env, "save"):
            vecnorm_dir = os.path.dirname(self.best_vecnorm_path)
            if vecnorm_dir:
                os.makedirs(vecnorm_dir, exist_ok=True)
            self.training_env.save(self.best_vecnorm_path)

        print(
            f"[{self.nn_type}] new best success={100 * self.best_success_rate:.1f}% "
            f"collision={100 * self.best_collision_rate:.1f}% "
            f"reward={self.best_reward_mean:.2f} "
            f"saved to {self.best_model_path}.zip",
            flush=True,
        )

    @staticmethod
    def _write_csv(path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _plot_all(self, df):
        plots = {
            "reward_mean": "Mean Reward",
            "length_mean": "Mean Episode Length",
            "success_rate": "Success Rate",
            "collision_rate": "Collision Rate",
        }
        for col, title in plots.items():
            self._plot(df, [col], title, f"{col}.png")
        self._plot(
            df,
            ["goals", "collisions", "timeouts"],
            "Goals vs Collisions vs Timeouts",
            "goals_collisions_timeouts.png",
        )
        self._plot(
            df,
            [f"action_{a}_rate" for a in range(self.n_actions)],
            "Action Distribution",
            "action_distribution.png",
        )

    def _plot(self, df, cols, title, filename):
        fig, ax = plt.subplots(figsize=(9, 5))
        for col in cols:
            if col in df:
                ax.plot(df["timestep"], df[col], label=col)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)
        if len(cols) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, filename), dpi=150)
        plt.close(fig)
