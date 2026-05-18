import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=128,
        num_actions=6,
        embed_dim=8,
        hidden_size=64,
        num_layers=1,
    ):
        super().__init__(observation_space, features_dim)

        self.lidar_dim = observation_space.shape[0] - 1
        self.num_actions = num_actions

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.embedding = nn.Embedding(num_actions + 1, embed_dim)

        self.linear = nn.Sequential(
            nn.Linear(hidden_size + embed_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Assuming observation array is:
        observations[:lidar_dim] --> LiDAR sequence
        observations[lidar_dim] --> previous action
        """
        lidar = observations[:, :self.lidar_dim]
        prev_action = observations[:, self.lidar_dim].long()

        _, (hidden, _) = self.lstm(lidar.unsqueeze(-1))
        x = hidden[-1]

        prev_action_embedded_idx = torch.where(
            prev_action == -1, self.num_actions, prev_action
        )
        prev_action_embedded_idx = torch.clamp(
            prev_action_embedded_idx, 0, self.num_actions
        )
        embedded = self.embedding(prev_action_embedded_idx)

        combined = torch.cat([x, embedded], dim=1)
        return self.linear(combined)
