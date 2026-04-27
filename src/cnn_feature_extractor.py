import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, num_actions=6, embed_dim=8):
        super().__init__(observation_space, features_dim)

        self.lidar_dim = observation_space.shape[0] - 1

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_lidar = torch.zeros((1, 1, self.lidar_dim), dtype=torch.float32)
            cnn_output_dim = self.cnn(sample_lidar).shape[1]

        self.embedding = nn.Embedding(num_actions + 1, embed_dim)
        self.num_actions = num_actions

        total_input_dim = cnn_output_dim + embed_dim

        self.linear = nn.Sequential(
            nn.Linear(total_input_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Assuming observation array is:
        observations[:lidar_dim] --> LiDAR
        observations[lidar_dim] --> previous action
        """
        lidar = observations[:, :self.lidar_dim]
        prev_action = observations[:, self.lidar_dim].long()

        x = self.cnn(lidar.unsqueeze(1))

        prev_action_embedded_idx = torch.where(
            prev_action == -1, self.num_actions, prev_action
        )

        prev_action_embedded_idx = torch.clamp(
            prev_action_embedded_idx, 0, self.num_actions
        )

        embedded = self.embedding(prev_action_embedded_idx)

        combined = torch.cat([x, embedded], dim=1)

        return self.linear(combined)
