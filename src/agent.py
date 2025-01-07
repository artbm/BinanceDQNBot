from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque, namedtuple
import os
import json
import logging
from datetime import datetime
from dataclasses import dataclass
import threading
from .utils.logger import get_logger

logger = get_logger(__name__)

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class AgentConfig:
    batch_size: int
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    learning_rate: float
    target_update_freq: int
    memory_size: int
    min_memory_size: int
    hidden_size: List[int]
    model_path: str
    save_freq: int

class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        """
        Deep Q-Network architecture.
        
        Args:
            input_size: Size of input features
            output_size: Number of actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Prioritized Experience Replay buffer.
        
        Args:
            capacity: Maximum size of buffer
            alpha: Priority exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.lock = threading.Lock()
        
    def add(self, experience: Experience, error: float = None):
        """Add experience to buffer with priority."""
        with self.lock:
            priority = max(self.priorities) if self.priorities else 1.0
            if error is not None:
                priority = (abs(error) + 1e-5) ** self.alpha
            
            self.buffer.append(experience)
            self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch of experiences based on priorities."""
        with self.lock:
            total_priority = sum(self.priorities)
            probabilities = np.array(self.priorities) / total_priority
            
            # Sample indices based on priorities
            indices = np.random.choice(
                len(self.buffer),
                batch_size,
                p=probabilities,
                replace=False
            )
            
            # Calculate importance sampling weights
            weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
            weights = weights / weights.max()
            
            experiences = [self.buffer[idx] for idx in indices]
            
            return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities for experiences."""
        with self.lock:
            for idx, error in zip(indices, errors):
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: AgentConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Input state dimension
            action_size: Number of possible actions
            config: Agent configuration
            device: Device to run the model on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = device
        
        # Initialize networks
        self.policy_net = DQNNetwork(
            state_size,
            action_size,
            config.hidden_size
        ).to(device)
        
        self.target_net = DQNNetwork(
            state_size,
            action_size,
            config.hidden_size
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize memory
        self.memory = PrioritizedReplayBuffer(config.memory_size)
        
        # Initialize training variables
        self.epsilon = config.epsilon_start
        self.steps = 0
        self.episodes = 0
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(f'runs/dqn_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Load model if exists
        self.load_model()

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        # Calculate TD error for prioritized replay
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            current_q = self.policy_net(state_tensor)[0, action]
            next_q = self.target_net(next_state_tensor).max()
            expected_q = reward + (1 - done) * self.config.gamma * next_q
            
            error = abs(current_q - expected_q).item()
        
        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience, error)

    def replay(self) -> Optional[float]:
        """Train on batch of experiences."""
        if len(self.memory) < self.config.min_memory_size:
            return None
            
        try:
            # Sample batch of experiences
            experiences, indices, weights = self.memory.sample(self.config.batch_size)
            
            # Prepare batch tensors
            states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
            actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
            dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # Get current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Get next Q values from target network
            with torch.no_grad():
                # Double DQN
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                
            # Calculate target Q values
            target_q_values = rewards.unsqueeze(1) + \
                            (1 - dones.unsqueeze(1)) * self.config.gamma * next_q_values
            
            # Calculate loss with importance sampling weights
            td_errors = target_q_values - current_q_values
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
            
            # Update priorities
            self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
            
            # Optimize the policy network
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update target network if needed
            if self.steps % self.config.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Update epsilon
            self.epsilon = max(
                self.config.epsilon_min,
                self.epsilon * self.config.epsilon_decay
            )
            
            # Log metrics
            self.writer.add_scalar('Loss/train', loss.item(), self.steps)
            self.writer.add_scalar('Epsilon', self.epsilon, self.steps)
            
            self.steps += 1
            
            # Save model periodically
            if self.steps % self.config.save_freq == 0:
                self.save_model()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in replay: {e}")
            return None

    def save_model(self):
        """Save model and training state."""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            
            # Save model state
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'steps': self.steps,
                'episodes': self.episodes,
                'epsilon': self.epsilon
            }, self.config.model_path)
            
            logger.info(f"Model saved to {self.config.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self):
        """Load model and training state."""
        try:
            if os.path.exists(self.config.model_path):
                checkpoint = torch.load(self.config.model_path)
                
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.steps = checkpoint['steps']
                self.episodes = checkpoint['episodes']
                self.epsilon = checkpoint['epsilon']
                
                logger.info(f"Model loaded from {self.config.model_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def update_learning_rate(self, new_lr: float):
        """Update learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_model_summary(self) -> Dict:
        """Get model training summary."""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def cleanup(self):
        """Cleanup resources."""
        self.writer.close()
        self.save_model()
