# rl-webots

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
