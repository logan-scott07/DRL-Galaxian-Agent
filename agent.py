from collections import deque
from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn as nn


class AgentNN(nn.Module):
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

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool

class ReplayMemory:
    def __init__(self, maxlen: int):
        self.memory = deque([], maxlen=maxlen)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, sample_size: int) -> Experience:
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent:
    def __init__(self, n_actions, state_shape, device, buffer_capacity=10_000):
        self.n_actions = n_actions
        self.device = device
        self.policy_net = AgentNN(n_actions, in_channels=state_shape[0]).to(device)
        self.memory = ReplayMemory(buffer_capacity)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            state_tensor = torch.tensor(state, device=self.device).float().unsqueeze(0)
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax().item()
        return action

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.push(Experience(state, action, reward, next_state, terminated, truncated))

    #def train(self):

    def memory_size(self) -> int:
        return len(self.memory)