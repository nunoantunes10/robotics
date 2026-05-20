import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, num_actions=6):
        super().__init__(observation_space, features_dim)

        self.context_dim = 3 + num_actions
        self.lidar_dim = observation_space.shape[0] - self.context_dim
        if self.lidar_dim <= 0:
            raise ValueError("Observation space is too small for LiDAR plus context")

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

        total_input_dim = cnn_output_dim + self.context_dim

        self.linear = nn.Sequential(
            nn.Linear(total_input_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Assuming observation array is:
        observations[:lidar_dim] --> normalized LiDAR
        observations[lidar_dim:] --> goal context and previous action one-hot
        """
        lidar = observations[:, :self.lidar_dim]
        context = observations[:, self.lidar_dim:]

        x = self.cnn(lidar.unsqueeze(1))
        combined = torch.cat([x, context], dim=1)

        return self.linear(combined)
