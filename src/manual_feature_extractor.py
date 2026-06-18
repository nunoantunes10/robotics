import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ManualFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, num_actions=6, embed_dim=8, num_sectors=36):
        super().__init__(observation_space, features_dim)
        
        self.lidar_dim = observation_space.shape[0] - 1
        self.num_sectors = num_sectors
        
        # If lidar_dim is not divisible by num_sectors, we'll pad or truncate.
        # Assuming lidar_dim is typically 360 and num_sectors is 36.
        self.points_per_sector = self.lidar_dim // self.num_sectors
        
        self.embedding = nn.Embedding(num_actions + 1, embed_dim)
        self.num_actions = num_actions
        
        # For each sector we extract: min, mean, and variance
        manual_feature_dim = self.num_sectors * 3
        total_input_dim = manual_feature_dim + embed_dim
        
        self.linear = nn.Sequential(
            nn.Linear(total_input_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lidar = observations[:, :self.lidar_dim]
        prev_action = observations[:, self.lidar_dim].long()
        batch_size = lidar.shape[0]
        
        # Reshape to (batch_size, num_sectors, points_per_sector)
        # Handle cases where it doesn't divide perfectly
        if self.lidar_dim % self.num_sectors != 0:
            usable_dim = self.points_per_sector * self.num_sectors
            lidar = lidar[:, :usable_dim]
            
        sectors = lidar.view(batch_size, self.num_sectors, self.points_per_sector)
        
        # Min distance (closest obstacle in sector)
        sector_mins, _ = torch.min(sectors, dim=2)
        
        # Mean distance (openness of the sector)
        sector_means = torch.mean(sectors, dim=2)
        
        # Variance (roughness/wall detection)
        sector_vars = torch.var(sectors, dim=2, unbiased=False)
        
        manual_features = torch.cat([sector_mins, sector_means, sector_vars], dim=1)
        
        prev_action_embedded_idx = torch.where(
            prev_action == -1, self.num_actions, prev_action
        )
        prev_action_embedded_idx = torch.clamp(
            prev_action_embedded_idx, 0, self.num_actions
        )
        
        embedded = self.embedding(prev_action_embedded_idx)
        
        combined = torch.cat([manual_features, embedded], dim=1)
        
        return self.linear(combined)
