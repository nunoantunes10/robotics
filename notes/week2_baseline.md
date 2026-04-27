# Week 2 Baseline

## Branch
week2-baseline

## Current working setup
- Webots world: worlds/smart-wheelchairs.wbt
- Training script: src/rl-server.py
- Test script: src/rl-test.py
- Robot controller: controllers/robot_client/robot_client.py

## Smoke test settings
- TRAIN_STEPS = 2048
- N_ROBOTS (train) = 9
- N_ROBOTS (test) = 9
- device = cpu
- Vec env = DummyVecEnv
- PPO n_steps = 128
- Test model path = ./models/ppo_wheelchair

## Smoke test result
- Model file created: models/ppo_wheelchair.zip
- Success file created: success_rates.csv
- Observed result: 0% success for all 9 robots
- Important note: pipeline runs end-to-end successfully

## Manual patches applied
- Reduced training/test budget for smoke test
- Changed rl-test.py model path to ./models/ppo_wheelchair
- Removed forced sys.exit(0) from robot_client.py
- Redirected trajectory saving to local logs/ folder

## Instrumented baseline test summary
- episodes = 47
- success_rate_pct = 0.0
- collision_rate_pct = 100.0
- mean_episode_steps = 865.81
- mean_episode_reward = -3856.37
- max_episode_steps = 1425
- min_episode_steps = 12

## Baseline test after 20000 training steps
- episodes = 48
- success_rate_pct = 0.0
- collision_rate_pct = 100.0
- mean_episode_steps = 837.65
- mean_episode_reward = -3657.31
- max_episode_steps = 2143
- min_episode_steps = 1
- note = slight improvement in reward/steps, but still no successful episodes

## Lidar rays experiment: 90 vs 360
### lidar360
- episodes = 48
- success_rate_pct = 0.0
- collision_rate_pct = 100.0
- mean_episode_steps = 874.25
- mean_episode_reward = -3773.09
- max_episode_steps = 2060
- min_episode_steps = 1

### lidar90
- episodes = 48
- success_rate_pct = 0.0
- collision_rate_pct = 100.0
- mean_episode_steps = 852.96
- mean_episode_reward = -3790.01
- max_episode_steps = 1384
- min_episode_steps = 1

### quick_take
- Performance is very similar between 360 and 90 rays in this short training budget.
- lidar90 is slightly worse in mean reward.
