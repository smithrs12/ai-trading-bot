# reinforcement.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from config import config

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class PyTorchQLearningAgent:
    def __init__(self, state_size: int = 10, action_size: int = 3, lr: float = None,
                 epsilon=None, epsilon_decay=None, gamma=None):
        self.state_size = state_size
        self.action_size = action_size
        self.actions = ['buy', 'sell', 'hold']

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        # === Configurable Parameters ===
        self.epsilon = epsilon if epsilon is not None else config.Q_LEARNING_EPSILON
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else config.Q_LEARNING_EPSILON_DECAY
        self.gamma = gamma if gamma is not None else config.Q_LEARNING_GAMMA
        self.lr = lr if lr is not None else config.Q_LEARNING_LR

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        self.update_target_frequency = 100
        self.step_count = 0

        self.update_target_network()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, confidence=None):
        if confidence is not None and confidence >= 0.95:
            return 0  # Force 'buy'
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > config.Q_LEARNING_MIN_EPSILON:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)

    def load_model(self, filepath: str):
        try:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            print("✅ PyTorch Q-Network loaded.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    def reward_shaping(self, raw_pnl: float, drawdown: float = 0.0, regime: str = "neutral") -> float:
        """
        Custom reward function based on PnL, adjusted for risk and regime.
        """
        reward = raw_pnl
        if drawdown > 0.02:
            reward -= drawdown * 10  # Penalize excessive drawdown
        if regime == "bearish":
            reward *= 0.8  # Penalize aggressive trading in bear markets
        return reward
