# rl-webots

## Transfer comparison pipeline

`src/rl-experiment-pipeline.py` is being implemented in stages. Stage 6 creates
the run folder, validates static inputs, parses robot IDs from the Webots world
files, writes `config.json`, can launch Webots, can train selected models, and
can evaluate available models. It also generates summaries/reports/plots and can
optionally use early stopping with best-checkpoint saving.

Check the planned configuration:

```bash
python3 src/rl-experiment-pipeline.py \
  --train-world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --eval-world worlds/smart-wheelchairs-realworld-test.wbt \
  --output-root outputs/pipeline \
  --phase check
```

`--check-only` is an alias for `--phase check`:

```bash
python3 src/rl-experiment-pipeline.py \
  --train-world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --eval-world worlds/smart-wheelchairs-realworld-test.wbt \
  --output-root outputs/pipeline \
  --check-only
```

The staged interface is:

```bash
python3 src/rl-experiment-pipeline.py --phase check
python3 src/rl-experiment-pipeline.py --phase train
python3 src/rl-experiment-pipeline.py --phase eval
python3 src/rl-experiment-pipeline.py --phase report
python3 src/rl-experiment-pipeline.py --phase all
```

Resume a previous run folder for later stages:

```bash
python3 src/rl-experiment-pipeline.py \
  --resume-run outputs/pipeline/YYYYMMDD_HHMMSS_transfer_comparison \
  --phase report
```

Stage 6 training, evaluation, and reporting commands:

```bash
python3 src/rl-experiment-pipeline.py --phase train --launch-webots
python3 src/rl-experiment-pipeline.py --phase train --experiments scratch
python3 src/rl-experiment-pipeline.py --phase train --experiments transfer_finetune,transfer_goal_direction
python3 src/rl-experiment-pipeline.py --phase eval --launch-webots
python3 src/rl-experiment-pipeline.py --phase eval --resume-run outputs/pipeline/YYYYMMDD_HHMMSS_transfer_comparison
python3 src/rl-experiment-pipeline.py --phase eval --eval-checkpoint best --resume-run outputs/pipeline/YYYYMMDD_HHMMSS_transfer_comparison
python3 src/rl-experiment-pipeline.py --phase report --resume-run outputs/pipeline/YYYYMMDD_HHMMSS_transfer_comparison
python3 src/rl-experiment-pipeline.py --phase all --launch-webots
```

Long-running phases write timestamped progress messages to stdout and append them
to:

```text
outputs/pipeline/YYYYMMDD_HHMMSS_transfer_comparison/logs/progress.log
outputs/pipeline/YYYYMMDD_HHMMSS_transfer_comparison/<experiment>/logs/progress.log
```

Control the training/evaluation progress interval with `--progress-log-freq`:

```bash
python3 src/rl-experiment-pipeline.py \
  --phase train \
  --launch-webots \
  --progress-log-freq 10000
```

The train commands start real PPO training. The eval commands start real Webots
interaction and evaluate available models on
`worlds/smart-wheelchairs-realworld-test.wbt`. Use `--launch-webots` to let the
pipeline open and close worlds automatically, or open the required world manually
before running without `--launch-webots`.

When multiple experiments are selected with `--launch-webots`, the pipeline
restarts the training world for each experiment so robot clients are fresh after
the previous environment closes. Evaluation similarly restarts the evaluation
world per model in automatic mode. Without `--launch-webots`, reload the active
world between experiments/models if robot clients do not reconnect.

Early stopping is optional and only affects stopping/checkpointing on the active
training world. The official comparison still comes from evaluation on
`worlds/smart-wheelchairs-realworld-test.wbt`.

```bash
python3 src/rl-experiment-pipeline.py \
  --phase train \
  --launch-webots \
  --early-stop \
  --eval-freq 50000 \
  --early-stop-patience 5 \
  --early-stop-min-delta 1.0 \
  --early-stop-min-timesteps 200000
```

Tiny early-stopping smoke test only, not a real experiment:

```bash
python3 src/rl-experiment-pipeline.py \
  --phase train \
  --launch-webots \
  --experiments scratch \
  --max-timesteps 5000 \
  --early-stop \
  --eval-freq 1000 \
  --early-stop-eval-episodes 2 \
  --early-stop-patience 2 \
  --early-stop-min-timesteps 1000 \
  --overwrite
```

## Transfer fine-tuning

The transfer workflow reuses the existing PPO model and `VecNormalize` statistics,
then saves the adapted model under a new name. It does not launch Webots for you:
open `worlds/smart-wheelchairs-transfer-finetune.wbt` in Webots before running the
commands below.

Do not use `python3 src/rl-server.py --new` for transfer learning. The existing
training script deletes the previous model when `--new` is used.

### Evaluate the old model on the transfer map

```bash
python3 src/rl-test.py \
  --world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --model-path models/ppo_wheelchair_lidar360_human \
  --vecnormalize-path models/vecnormalize_lidar360_human.pkl
```

### Fine-tune on the transfer map

```bash
python3 src/rl-transfer.py \
  --world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --experiment-name transfer_finetune_lidar360 \
  --output-dir outputs/experiments \
  --source-model models/ppo_wheelchair_lidar360_human \
  --source-vecnormalize models/vecnormalize_lidar360_human.pkl \
  --output-model models/ppo_wheelchair_lidar360_human_transfer_finetune \
  --output-vecnormalize models/vecnormalize_lidar360_human_transfer_finetune.pkl \
  --learning-rate 1e-5 \
  --reward-mode transfer_finetune \
  --robots 9
```

The transfer script trains with `reset_num_timesteps=False`, refuses to save over
the source model/stat files, and refuses to replace existing output files unless
you pass `--overwrite-output`. The transfer reward mode adds conservative
straight-line shaping when the front LiDAR sector is clear, without changing the
normal training reward used by `src/rl-server.py`.

Each transfer run writes TensorBoard logs into its own experiment folder by
default:

```text
outputs/experiments/<experiment>/training/tensorboard/<timestamp>__ppo-lidar360-<reward_mode>/
```

This keeps older TensorBoard runs available when you fine-tune again or compare
different reward modes.

### Transfer reward: goal direction

This reward mode uses the robot's initial pose as a proxy for the goal direction.
It rewards progress along the initial global forward vector instead of rewarding
the forward action directly.

```bash
python3 src/rl-transfer.py \
  --world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --source-model models/ppo_wheelchair_lidar360_human \
  --source-vecnormalize models/vecnormalize_lidar360_human.pkl \
  --output-model models/ppo_wheelchair_lidar360_human_transfer_goal_direction \
  --output-vecnormalize models/vecnormalize_lidar360_human_transfer_goal_direction.pkl \
  --reward-mode transfer_goal_direction \
  --learning-rate 1e-5 \
  --robots 9
```

### Test the fine-tuned model

```bash
python3 src/rl-test.py \
  --world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --model-path models/ppo_wheelchair_lidar360_human_transfer_finetune \
  --vecnormalize-path models/vecnormalize_lidar360_human_transfer_finetune.pkl
```

Evaluation prints an action distribution so you can check whether fine-tuning is
increasing `forward` actions and reducing unnecessary left/right loops.

To save evaluation outputs into the experiment folder, pass `--run-dir`:

```bash
python3 src/rl-test.py \
  --world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --model-path models/ppo_wheelchair_lidar360_human \
  --vecnormalize-path models/vecnormalize_lidar360_human.pkl \
  --run-dir outputs/experiments/<experiment>/evaluation/old_model

python3 src/rl-test.py \
  --world worlds/smart-wheelchairs-transfer-finetune.wbt \
  --model-path models/ppo_wheelchair_lidar360_human_transfer_finetune \
  --vecnormalize-path models/vecnormalize_lidar360_human_transfer_finetune.pkl \
  --run-dir outputs/experiments/<experiment>/evaluation/finetuned_model
```

Generate the report after both evaluations:

```bash
python3 src/rl-report.py outputs/experiments/<experiment>
```

The report workflow writes comparison CSVs to `csv/`, plots to `plots/`, and a
single `report.html` with metadata, process parameters, evaluation metrics, action
distributions, time-to-target analysis, and links to raw artifacts.

For repeated real-world map tests with different models, prefer timestamped
evaluation runs with `--run-root` and `--run-label`:

```bash
python3 src/rl-test.py \
  --world worlds/smart-wheelchairs-realworld-test.wbt \
  --model-path models/ppo_wheelchair_lidar360_human_transfer_finetune \
  --vecnormalize-path models/vecnormalize_lidar360_human_transfer_finetune.pkl \
  --run-root outputs/experiments/realworld_test/evaluation \
  --run-label transfer_finetune \
  --timesteps 20000
```

This writes to a folder like
`outputs/experiments/realworld_test/evaluation/2026-06-16_12-30-00__smart-wheelchairs-realworld-test__transfer_finetune`.

### Checks before full training

- Confirm the transfer world launches correctly in Webots.
- Confirm all requested robot clients connect. The transfer world currently has
  robot IDs `0..8`, so the default `--robots 9` is valid.
- Keep the LiDAR dimension at `360`; the model expects observation shape `(361,)`
  because observations are `360` LiDAR values plus the previous action.
- Keep the action space unchanged at `Discrete(6)`.
- Keep the original model and original `VecNormalize` file as the baseline to
  avoid losing the pre-transfer policy.
