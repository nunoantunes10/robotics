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
