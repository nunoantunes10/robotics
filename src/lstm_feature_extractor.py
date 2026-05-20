import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Compatibility name for the old ``--nn lstm`` pipeline.

    This extractor is intentionally not recurrent. The previous implementation
    treated one LiDAR scan as an LSTM sequence and kept only the final hidden
    state, which made the policy sensitive to beam order and weak at preserving
    local obstacle/gap geometry. The replacement uses spatial LiDAR features:
    a small 1D CNN plus explicit sector min/mean distances.
    """

    def __init__(
        self,
        observation_space,
        features_dim=128,
        num_actions=6,
        num_sectors=12,
    ):
        super().__init__(observation_space, features_dim)

        self.context_dim = 3 + num_actions
        self.lidar_dim = observation_space.shape[0] - self.context_dim
        if self.lidar_dim <= 0:
            raise ValueError("Observation space is too small for LiDAR plus context")
        if num_sectors <= 0:
            raise ValueError("num_sectors must be positive")
        if num_sectors > self.lidar_dim:
            raise ValueError("num_sectors cannot exceed the LiDAR dimension")

        self.num_sectors = num_sectors
        self.sector_feature_dim = 2 * num_sectors
        self.cnn_output_dim = 64 * 16

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(
                self.cnn_output_dim + self.sector_feature_dim + self.context_dim,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Assuming observation array is:
        observations[:lidar_dim] --> normalized LiDAR scan
        observations[lidar_dim:] --> goal context and previous action one-hot
        """
        lidar = observations[:, :self.lidar_dim]
        context = observations[:, self.lidar_dim:]

        cnn_features = self.cnn(lidar.unsqueeze(1))
        sector_features = self._sector_features(lidar)

        combined = torch.cat([cnn_features, sector_features, context], dim=1)
        return self.linear(combined)

    def _sector_features(self, lidar: torch.Tensor) -> torch.Tensor:
        sectors = torch.tensor_split(lidar, self.num_sectors, dim=1)
        features = []
        for sector in sectors:
            features.append(torch.min(sector, dim=1).values)
            features.append(torch.mean(sector, dim=1))
        return torch.stack(features, dim=1)
