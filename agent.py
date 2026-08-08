import numpy as np
import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = 4 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        return self.net(x)