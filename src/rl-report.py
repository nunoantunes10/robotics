import argparse
import csv
import html
import json
import os
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/robotics-matplotlib-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


ACTION_LABELS = [
    "forward",
    "forward_left",
    "forward_right",
    "left",
    "right",
    "stop",
]


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def discover_evaluations(experiment_dir):
    evaluation_root = os.path.join(experiment_dir, "evaluation")
    if not os.path.isdir(evaluation_root):
        return []

    evaluations = []
    for name in sorted(os.listdir(evaluation_root)):
        run_dir = os.path.join(evaluation_root, name)
        if os.path.isdir(run_dir) and os.path.exists(os.path.join(run_dir, "metadata.json")):
            evaluations.append((name, run_dir))
    return evaluations


def summarize_evaluation(label, run_dir):
    summary_rows = load_csv_rows(os.path.join(run_dir, "run_summary.csv"))
    episode_rows = load_csv_rows(os.path.join(run_dir, "episode_metrics.csv"))
    metadata = read_json(os.path.join(run_dir, "metadata.json"))

    if summary_rows:
        summary = summary_rows[0]
    else:
        summary = {}

    total_episodes = as_int(summary.get("total_episodes"), len(episode_rows))
    successful_episodes = as_int(summary.get("successful_episodes"))
    success_rate = as_float(summary.get("success_rate"))
    avg_time_to_target = as_float(summary.get("avg_time_to_target"))
    min_time_to_target = as_float(summary.get("min_time_to_target"))
    max_time_to_target = as_float(summary.get("max_time_to_target"))
    completed_timesteps = as_int(summary.get("completed_timesteps"))

    collision_count = sum(as_int(row.get("collision")) for row in episode_rows)
    collision_rate = 100 * collision_count / total_episodes if total_episodes else 0.0
    avg_episode_steps = (
        sum(as_float(row.get("episode_steps")) for row in episode_rows) / len(episode_rows)
        if episode_rows
        else 0.0
    )
    avg_episode_reward = (
        sum(as_float(row.get("episode_reward")) for row in episode_rows) / len(episode_rows)
        if episode_rows
        else 0.0
    )

    return {
        "label": label,
        "run_dir": run_dir,
        "world": metadata.get("world_arg") or metadata.get("world") or summary.get("world", ""),
        "model": metadata.get("model_path") or summary.get("model", ""),
        "vecnormalize": metadata.get("vecnormalize_path", ""),
        "robots": metadata.get("robots", ""),
        "completed_timesteps": completed_timesteps,
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": round(success_rate, 2),
        "collision_count": collision_count,
        "collision_rate": round(collision_rate, 2),
        "avg_episode_steps": round(avg_episode_steps, 2),
        "avg_episode_reward": round(avg_episode_reward, 2),
        "avg_time_to_target": round(avg_time_to_target, 2),
        "min_time_to_target": round(min_time_to_target, 2),
        "max_time_to_target": round(max_time_to_target, 2),
    }


def load_action_distribution(label, run_dir):
    rows = load_csv_rows(os.path.join(run_dir, "action_distribution.csv"))
    if rows:
        return [
            {
                "label": label,
                "action": row.get("action", ""),
                "count": as_int(row.get("count")),
                "percent": as_float(row.get("percent")),
            }
            for row in rows
        ]

    metadata = read_json(os.path.join(run_dir, "metadata.json"))
    counts = metadata.get("action_counts", {})
    percents = metadata.get("action_distribution_percent", {})
    return [
        {
            "label": label,
            "action": action,
            "count": as_int(counts.get(action)),
            "percent": as_float(percents.get(action)),
        }
        for action in ACTION_LABELS
    ]


def plot_bar(rows, x_key, y_key, title, ylabel, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row[x_key]) for row in rows]
    values = [as_float(row[y_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#3267a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_action_distribution(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted({row["label"] for row in rows})
    x = list(range(len(ACTION_LABELS)))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, label in enumerate(labels):
        by_action = {
            row["action"]: as_float(row["percent"])
            for row in rows
            if row["label"] == label
        }
        values = [by_action.get(action, 0.0) for action in ACTION_LABELS]
        shifted = [pos + idx * width for pos in x]
        ax.bar(shifted, values, width=width, label=label)

    center_offset = width * (len(labels) - 1) / 2
    ax.set_xticks([pos + center_offset for pos in x])
    ax.set_xticklabels(ACTION_LABELS, rotation=30, ha="right")
    ax.set_title("Action Distribution by Model")
    ax.set_ylabel("Action share (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trajectory_overlays(evaluations, plots_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    outputs = []
    for label, run_dir in evaluations:
        trajectory_dir = os.path.join(run_dir, "trajectories")
        if not os.path.isdir(trajectory_dir):
            continue

        paths = [
            os.path.join(trajectory_dir, name)
            for name in sorted(os.listdir(trajectory_dir))
            if name.endswith(".csv")
        ]
        if not paths:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        plotted = 0
        for path in paths:
            df = pd.read_csv(path)
            if {"x", "y"}.issubset(df.columns):
                ax.plot(df["x"], df["y"], linewidth=1.3, alpha=0.75)
                plotted += 1
        if plotted:
            ax.set_title(f"Trajectory Overlay: {label}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            out_path = os.path.join(plots_dir, f"trajectory_overlay_{label}.png")
            fig.savefig(out_path, dpi=160)
            outputs.append(out_path)
        plt.close(fig)

    return outputs


def rel_link(base_dir, path):
    return os.path.relpath(path, base_dir).replace(os.sep, "/")


def table_html(rows, columns):
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def generate_html_report(
    experiment_dir,
    experiment_metadata,
    transfer_config,
    comparison_rows,
    action_rows,
    plot_paths,
):
    report_path = os.path.join(experiment_dir, "report.html")
    generated_at = datetime.now().isoformat(timespec="seconds")

    comparison_columns = [
        ("label", "Run"),
        ("success_rate", "Success %"),
        ("collision_rate", "Collision %"),
        ("total_episodes", "Episodes"),
        ("avg_time_to_target", "Avg Time to Target"),
        ("min_time_to_target", "Min Time to Target"),
        ("max_time_to_target", "Max Time to Target"),
        ("avg_episode_steps", "Avg Episode Steps"),
        ("avg_episode_reward", "Avg Episode Reward"),
        ("completed_timesteps", "Eval Timesteps"),
    ]
    action_columns = [
        ("label", "Run"),
        ("action", "Action"),
        ("count", "Count"),
        ("percent", "Percent"),
    ]

    raw_links = []
    for root, _, files in os.walk(experiment_dir):
        for filename in sorted(files):
            if filename.endswith((".csv", ".json")):
                path = os.path.join(root, filename)
                raw_links.append(path)

    plot_html = "\n".join(
        f'<figure><img src="{html.escape(rel_link(experiment_dir, path))}" '
        f'alt="{html.escape(os.path.basename(path))}"><figcaption>{html.escape(os.path.basename(path))}</figcaption></figure>'
        for path in plot_paths
    )
    raw_link_html = "\n".join(
        f'<li><a href="{html.escape(rel_link(experiment_dir, path))}">{html.escape(rel_link(experiment_dir, path))}</a></li>'
        for path in raw_links
    )

    config_items = {
        "Experiment": experiment_metadata.get("experiment_name", ""),
        "World": experiment_metadata.get("world", ""),
        "Source model": experiment_metadata.get("source_model", ""),
        "Fine-tuned model": experiment_metadata.get("finetuned_model", ""),
        "Source VecNormalize": experiment_metadata.get("source_vecnormalize", ""),
        "Fine-tuned VecNormalize": experiment_metadata.get("finetuned_vecnormalize", ""),
        "Reward mode": transfer_config.get("reward_mode", ""),
        "Learning rate": transfer_config.get("learning_rate", ""),
        "Training timesteps": transfer_config.get("timesteps", ""),
        "Generated at": generated_at,
    }
    config_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in config_items.items()
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RL Transfer Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    figure {{ margin: 18px 0 28px; }}
    figcaption {{ color: #52606d; font-size: 0.9rem; margin-top: 6px; }}
    code {{ background: #f1f5f9; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>RL Transfer Fine-Tuning Report</h1>
  <h2>Experiment Metadata</h2>
  <table><tbody>{config_rows}</tbody></table>
  <h2>Evaluation Comparison</h2>
  {table_html(comparison_rows, comparison_columns)}
  <h2>Action Distribution</h2>
  {table_html(action_rows, action_columns)}
  <h2>Plots</h2>
  {plot_html or '<p>No plots were generated.</p>'}
  <h2>Raw Artifacts</h2>
  <ul>{raw_link_html}</ul>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return report_path


def generate_report(experiment_dir):
    experiment_dir = os.path.abspath(experiment_dir)
    csv_dir = os.path.join(experiment_dir, "csv")
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    experiment_metadata = read_json(
        os.path.join(experiment_dir, "metadata", "experiment.json")
    )
    transfer_config = read_json(
        os.path.join(experiment_dir, "training", "transfer_config.json")
    )
    evaluations = discover_evaluations(experiment_dir)
    if not evaluations:
        raise FileNotFoundError(
            f"No evaluation runs found under {os.path.join(experiment_dir, 'evaluation')}"
        )

    comparison_rows = [
        summarize_evaluation(label, run_dir) for label, run_dir in evaluations
    ]
    action_rows = [
        row
        for label, run_dir in evaluations
        for row in load_action_distribution(label, run_dir)
    ]

    comparison_csv = os.path.join(csv_dir, "evaluation_comparison.csv")
    write_csv(
        comparison_csv,
        comparison_rows,
        [
            "label",
            "world",
            "model",
            "vecnormalize",
            "robots",
            "completed_timesteps",
            "total_episodes",
            "successful_episodes",
            "success_rate",
            "collision_count",
            "collision_rate",
            "avg_episode_steps",
            "avg_episode_reward",
            "avg_time_to_target",
            "min_time_to_target",
            "max_time_to_target",
            "run_dir",
        ],
    )

    action_csv = os.path.join(csv_dir, "action_distribution_comparison.csv")
    write_csv(action_csv, action_rows, ["label", "action", "count", "percent"])

    plot_paths = []
    plot_specs = [
        ("success_rate", "Success Rate by Model", "Success (%)", "success_rate_by_model.png"),
        ("collision_rate", "Collision Rate by Model", "Collision (%)", "collision_rate_by_model.png"),
        ("avg_episode_steps", "Average Episode Steps by Model", "Steps", "avg_episode_steps_by_model.png"),
        ("avg_time_to_target", "Average Time to Target by Model", "Steps", "time_to_target_by_model.png"),
    ]
    for metric, title, ylabel, filename in plot_specs:
        path = os.path.join(plots_dir, filename)
        plot_bar(comparison_rows, "label", metric, title, ylabel, path)
        plot_paths.append(path)

    action_plot = os.path.join(plots_dir, "action_distribution_by_model.png")
    plot_action_distribution(action_rows, action_plot)
    plot_paths.append(action_plot)
    plot_paths.extend(plot_trajectory_overlays(evaluations, plots_dir))

    report_path = generate_html_report(
        experiment_dir,
        experiment_metadata,
        transfer_config,
        comparison_rows,
        action_rows,
        plot_paths,
    )
    print(f"Wrote comparison CSV to {comparison_csv}")
    print(f"Wrote action distribution CSV to {action_csv}")
    print(f"Wrote report to {report_path}")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CSV summaries, plots, and an HTML report for an RL experiment."
    )
    parser.add_argument("experiment_dir", help="Path to outputs/experiments/<experiment>")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        generate_report(parse_args().experiment_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
