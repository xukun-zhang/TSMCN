import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, num_points):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(num_points, num_points // 2),
            nn.ReLU(),
            nn.Linear(num_points // 2, num_points),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, x.size(1))
        return x

class CBAMModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, num_points=1024):
        super(CBAMModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(in_channels, num_points)

    def forward(self, x):
        x = self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Example usage
in_channels = 64  # Example feature channels
B, C, N = 10, in_channels, 10240  # Example sizes
features = torch.rand(B, C, N)  # Example input tensor

cbam_module = CBAMModule(in_channels, num_points=N)
output_features = cbam_module(features)
print(output_features.shape)
