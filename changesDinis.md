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

Current reward values:

- Time penalty per non-terminal step: `-0.01`
- Goal progress reward: `+10.0 * progress`, where `progress = previous_goal_distance - current_goal_distance`
- Forward action bonus: `+0.05`
- Rotate/stop action penalty: `-0.02`
- Front lidar danger penalty when `min_front < 1.0`: `-0.25`
- Front lidar danger penalty when `min_front < 0.5`: additional `-1.0`
- Collision terminal reward: `-500.0`
- Goal terminal reward: `1000.0`
