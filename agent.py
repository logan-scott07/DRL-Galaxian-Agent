from collections import deque
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import optim
import torch.nn as nn

E_START = 1.0
E_END = 0.01
E_DECAY = 9000
BATCH_SIZE = 1000
LR = 3e-4
GAMMA = 0.99

class DQN(nn.Module):
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
    def __init__(self, n_actions, state_shape, device, batch_size):
        self.n_actions = n_actions
        self.device = device

        self.policy_net = DQN(n_actions, in_channels=state_shape[0]).to(device)
        self.target_net = DQN(n_actions, in_channels=state_shape[0]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.batch_size = BATCH_SIZE
        self.memory = ReplayMemory(batch_size)
        self.step = 0

    def act(self, state: np.ndarray) -> int:
        epsilon = self.get_epsilon()
        self.step+=1
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            state_tensor = torch.tensor(state, device=self.device).float().unsqueeze(0)
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax().item()
        return action

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.push(Experience(state, action, reward, next_state, terminated, truncated))

    def train(self):
        if self.memory_size() < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)

        states = np.stack([e.state for e in batch])
        actions = [e.action for e in batch]
        rewards = [e.reward for e in batch]
        next_states = np.stack([e.next_state for e in batch])
        terminated = [e.terminated for e in batch]

        states = torch.tensor(states, device=self.device).float()
        next_states = torch.tensor(next_states, device=self.device).float()
        actions = torch.tensor(actions, device=self.device).long().unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device).float()
        terminated = torch.tensor(terminated, device=self.device).float()

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1).values
            target = rewards + (1 - terminated) * GAMMA * next_q_max

        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def memory_size(self) -> int:
        return len(self.memory)

    def get_epsilon(self) -> float:
        return E_END + (E_START - E_END) * np.exp(-self.step / E_DECAY)