# Changes by Dinis

## Reward redesign

- Reward is now computed from the post-action observation, matching the transition `state_t + action_t -> state_{t+1}, reward_t`.
- Collision and goal terminal rewards now dominate the return: collisions return `-500.0`, while goal completion returns `1000.0`.
- Dense rewards were reduced to a small time penalty, a small forward-motion bonus, optional goal-distance progress, and bounded front-lidar danger penalties.
- Goal distance is computed in the Webots robot client for reward shaping only; it is not added to the policy observation vector.
- Existing training/test flow, VecNormalize handling, model paths, logging format, and custom metrics are preserved.

## Goal distance and reward tuning

- Goal distance is now computed from the robot position sampled after the action is applied, avoiding a one-tick stale progress signal.
- Reward shaping now weights distance progress more strongly, rewards forward motion slightly more, and penalizes rotate/stop actions when they do not make progress.
- Front-lidar danger penalties were reduced and bounded so they remain useful without overwhelming long episodes.
- Step `info` now includes `goal_distance`, `progress`, and `min_front` diagnostics for checking terminal distances in Webots runs.

Reward values after this change:

- Time penalty per non-terminal step: `-0.01`
- Goal progress reward: `+10.0 * progress`, where `progress = previous_goal_distance - current_goal_distance`
- Forward action bonus: `+0.05`
- Rotate/stop action penalty: `-0.02`
- Front lidar danger penalty when `min_front < 1.0`: `-0.25`
- Front lidar danger penalty when `min_front < 0.5`: additional `-1.0`
- Collision terminal reward: `-500.0`
- Goal terminal reward: `1000.0`

## Obstacle avoidance reward update

- After observing robots driving too directly into obstacles and failing corner turns, forward reward is now only given when front clearance is safe.
- Progress reward is reduced near close front obstacles so the robot is not rewarded strongly for taking a short path through walls.
- Added side-clearance diagnostics and small turn-shaping rewards to help the robot learn to turn toward open space at corners.

Reward values after this change:

- Time penalty per non-terminal step: `-0.01`
- Goal progress reward: normally `+8.0 * progress`, where `progress = previous_goal_distance - current_goal_distance`
- Goal progress reward near obstacles: positive progress is reduced to `+2.0 * progress` when `min_front < 0.7`
- Safe forward action bonus when `min_front >= 1.2`: `+0.03`
- Forward-near-obstacle penalty when `min_front < 1.2`: `-0.50`
- Rotate/stop action penalty: `-0.01`
- Front lidar danger penalty when `min_front < 1.0`: `-0.75`
- Front lidar danger penalty when `min_front < 0.5`: additional `-2.0`
- Front lidar danger penalty when `min_front < 0.3`: additional `-3.0`
- Corner/obstacle turn reward when `min_front < 1.2` and turning toward the clearer side: `+0.25`
- Corner/obstacle wrong-turn penalty when `min_front < 1.2` and turning away from the clearer side: `-0.25`
- Ambiguous obstacle turn reward when both sides are similar and the robot turns: `+0.05`
- Collision terminal reward: `-500.0`
- Goal terminal reward: `1000.0`

## Reduced unnecessary turning reward update

- After observing the success rate decline and simple-path robots turning for no reason, removed the side-clearance turn reward from the active reward.
- Forward motion is now preferred when the front is clear; turning is only lightly rewarded when the robot is actually close to a front obstacle.
- Goal terminal reward was increased so successful episodes remain clearly better even if the path has accumulated dense penalties.

Reward values after this change:

- Time penalty per non-terminal step: `-0.01`
- Goal progress reward: normally `+8.0 * progress`, where `progress = previous_goal_distance - current_goal_distance`
- Goal progress reward near obstacles: positive progress is reduced to `+2.0 * progress` when `min_front < 0.6`
- Safe forward action bonus when `min_front >= 1.0`: `+0.06`
- Forward-near-obstacle penalty when `min_front < 1.0`: `-0.40`
- Turn action reward when `min_front < 0.8`: `+0.04`
- Rotate/stop penalty when front is not blocked: `-0.04`
- Front lidar danger penalty when `min_front < 0.8`: `-0.50`
- Front lidar danger penalty when `min_front < 0.5`: additional `-1.50`
- Front lidar danger penalty when `min_front < 0.3`: additional `-3.0`
- Collision terminal reward: `-500.0`
- Goal terminal reward: `2000.0`

## Progress-conditioned turning update

- Turning is not treated as bad by itself; it is rewarded when the post-action transition gets the robot closer to the goal.
- Forward-left and forward-right actions keep the normal forward reward and receive an extra bonus when they make positive progress.
- Turning without progress is still discouraged when the front is clear, to reduce pointless spinning in simple corridors.
- The previous goal terminal reward of `2000.0` is kept.

Reward values after this change:

- Time penalty per non-terminal step: `-0.01`
- Goal progress reward: normally `+8.0 * progress`, where `progress = previous_goal_distance - current_goal_distance`
- Goal progress reward near obstacles: positive progress is reduced to `+2.0 * progress` when `min_front < 0.6`
- Safe forward action bonus when `min_front >= 1.0`: `+0.06`
- Forward-near-obstacle penalty when `min_front < 1.0`: `-0.40`
- Forward-turn bonus when `progress > 0`: `+0.04`
- Forward-turn penalty when front is clear and `progress <= 0`: `-0.02`
- Pure turn bonus when `progress > 0`: `+0.03`
- Pure turn bonus when front is blocked with `min_front < 0.8`: `+0.03`
- Rotate/stop penalty otherwise: `-0.04`
- Front lidar danger penalty when `min_front < 0.8`: `-0.50`
- Front lidar danger penalty when `min_front < 0.5`: additional `-1.50`
- Front lidar danger penalty when `min_front < 0.3`: additional `-3.0`
- Collision terminal reward: `-500.0`
- Goal terminal reward: `2000.0`

## Goal-aware observation and PPO stability update

- The policy observation now includes normalized LiDAR, clipped goal distance, goal-bearing sine/cosine, and a one-hot previous action.
- Reset no longer returns an all-zero fake observation; it asks Webots for a real reset observation before the next episode starts.
- Removed the integer previous-action embedding from the feature extractors because `VecNormalize` can change the scalar action value before it is cast to an embedding index.
- Observation normalization is now manual in the environment, and `VecNormalize` is kept with `norm_obs=False` and `norm_reward=False`.
- New goal-aware model paths are used so incompatible old checkpoints are not loaded by accident.
- The CNN is the recommended model for this fix; the current LSTM extractor scans the LiDAR ray sequence but is not recurrent PPO memory across timesteps.
- A best-success checkpoint is saved during training and `rl-test.py` prefers it, reducing the chance that a later PPO update hides the best learned behavior.
- The old side-clearance `navigation_reward()` remains inactive because it can prefer the open side even when the goal is on the other branch.

Reward/training values after this change:

- Observation layout: `lidar/10`, `clip(goal_distance / 10, 0, 1)`, `sin(goal_bearing)`, `cos(goal_bearing)`, and 6 previous-action one-hot values.
- Time penalty per non-terminal step: `-0.005`
- Goal progress reward: normally `+6.0 * clipped_progress`, where `clipped_progress = clip(previous_goal_distance - current_goal_distance, -0.25, 0.25)`
- Goal progress reward near obstacles: `+2.0 * clipped_progress` when `min_front < 0.6`
- Safe forward action bonus when `min_front >= 1.0`: `+0.03`
- Forward-near-obstacle penalty when `min_front < 1.0`: `-0.25`
- Forward-turn progress bonus when `progress > 0`: `+0.02`
- Pure turn bonus when front is blocked with `min_front < 0.8`: `+0.02`
- Rotate/stop penalty otherwise: `-0.03`
- Front lidar danger penalty when `min_front < 0.8`: `-0.25`
- Front lidar danger penalty when `min_front < 0.5`: additional `-0.75`
- Front lidar danger penalty when `min_front < 0.3`: additional `-1.50`
- Collision terminal reward: `-50.0`
- Goal terminal reward: `100.0`
- PPO `n_steps`: `512`
- PPO `batch_size`: `1024`
- PPO `n_epochs`: `8`
- PPO `learning_rate`: `3e-5`
- PPO `clip_range`: `0.08`
- PPO `ent_coef`: `0.003`
- PPO `gamma`: `0.995`
- PPO `gae_lambda`: `0.95`
- PPO `target_kl`: `0.02`

## LSTM training stop stability hotfix

- After the goal-aware update, short LSTM runs were ending with no completed episodes in the metrics file, so the environment now truncates very long episodes at `3000` steps.
- `WheelchairEnv.reset()` now sends an explicit reset command to the Webots controller instead of sending a normal stop action, avoiding socket handshakes that can block when Gym resets a truncated episode.
- The Webots robot client now handles reset and terminate commands separately from motor actions and replies with a fresh observation after each reset.
- After collision or goal, the robot client now resets before sending the terminal reply, and the environment reuses that post-reset observation when SB3 immediately calls `reset()`.
- Automatic post-episode resets no longer send a second reset command, avoiding the observed `Env 2 timed out waiting for reply to [-2]` failure.
- Active reset commands were removed from the normal training reset path after repeated `[-2]` timeouts; reset now reuses the cached post-reset observation, the latest known observation, or the zero bootstrap observation without touching the socket.
- Environment sockets now use send/receive timeouts and report the `env_id` plus last request when a controller stops replying.
- Cleanup no longer sends a terminate command when a REQ socket is already waiting for a reply, avoiding the ZeroMQ `Operation cannot be accomplished in current state` error after Ctrl-C.
- Timeout/truncation episodes are logged separately from collisions and goals so they do not look like successful navigation.
- The LSTM training profile now uses smaller rollouts and update batches than the CNN profile, reducing the chance of unstable or very slow updates with the scan-LSTM extractor.
- Goal-aware model paths were moved to `_v2` names so short checkpoints saved before this hotfix are not loaded accidentally.
- The training script now prints the target timestep count and logs a traceback before saving if training stops because of an exception.

Training values after this hotfix:

- Environment time limit: `3000` steps per episode
- Environment socket timeout: `10000 ms`
- CNN PPO values remain: `n_steps=512`, `batch_size=1024`, `n_epochs=8`, `learning_rate=3e-5`, `clip_range=0.08`, `target_kl=0.02`
- LSTM PPO values: `n_steps=128`, `batch_size=512`, `n_epochs=4`, `learning_rate=2e-5`, `clip_range=0.05`, `target_kl=0.03`

## LSTM pipeline replacement with hybrid LiDAR PPO

- The old `--nn lstm` extractor has been replaced because it was not recurrent PPO memory across timesteps; it treated the 360 LiDAR beams as one sequence and kept only the final hidden state.
- The replacement keeps the `lstm` CLI name for compatibility but uses a hybrid LiDAR extractor: 1D CNN scan features plus 12 sector min/mean distance features and the existing goal/action context.
- No `sb3-contrib` or true recurrent PPO dependency was added in this step because the installed environment only includes Stable-Baselines3 PPO, and the immediate instability was the scan feature extractor.
- LSTM model paths were moved to `_goal_v3_` names so older v2 scan-LSTM checkpoints are not loaded into the new architecture.
- Best-checkpoint saving now breaks equal-success ties using lower collision rate and then higher mean reward, preserving better policies when success rate plateaus.
- Timeout episodes now receive a compact terminal penalty so long wandering episodes produce a clearer learning signal.

Reward/training values after this change:

- LSTM observation handling remains: normalized LiDAR, clipped goal distance, goal-bearing sine/cosine, and 6 previous-action one-hot values.
- Hybrid LiDAR sectors: `12` sectors, each contributing minimum and mean normalized distance.
- Timeout terminal reward: `-25.0`
- Collision terminal reward remains: `-50.0`
- Goal terminal reward remains: `100.0`
- LSTM PPO `n_steps`: `256`
- LSTM PPO `batch_size`: `512`
- LSTM PPO `n_epochs`: `6`
- LSTM PPO `learning_rate`: `2.5e-5`
- LSTM PPO `clip_range`: `0.05`
- LSTM PPO `ent_coef`: `0.001`
- LSTM PPO `gamma`: `0.995`
- LSTM PPO `gae_lambda`: `0.95`
- LSTM PPO `target_kl`: `0.02`
