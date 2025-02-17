import os
import numpy as np
import gymnasium as gym
from collections import deque
import random

import ale_py

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# DQN Network for RAM input
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN Network
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = ReplayBuffer(10000)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, episode, reward):
        """Save model checkpoint and training metrics"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(self.checkpoint_dir, f'dqn_checkpoint_ep{episode}_{timestamp}.pt')
        metrics_path = os.path.join(self.checkpoint_dir, f'training_metrics_ep{episode}_{timestamp}.json')
        
        # Save model state
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'reward': reward
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            
            print(f"Loaded checkpoint from episode {checkpoint['episode']} with reward {checkpoint['reward']}")
            return checkpoint['episode']
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return 0

# Environment setup and training
def main():
    gym.register_envs(ale_py)

    # Create environment without wrapper
    env = gym.make("ALE/VideoChess-v5", render_mode="human", obs_type="ram")
    
    state_size = env.observation_space.shape[0]  # RAM state size (should be 128 for Atari)
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")

    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Configuration
    episodes = 1000
    update_target_every = 100
    save_checkpoint_every = 50  # Save every 50 episodes
    
    # Load checkpoint if exists
    last_checkpoint = "checkpoints/last_checkpoint.pt"
    start_episode = agent.load_checkpoint(last_checkpoint) if os.path.exists(last_checkpoint) else 0
    
    # Training loop
    for episode in range(start_episode, episodes):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            
        # Update metrics
        agent.update_metrics(episode + 1, total_reward)
        
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_network()
        
        # Save checkpoint periodically
        if episode % save_checkpoint_every == 0:
            agent.save_checkpoint(episode + 1, total_reward)
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    # Save final checkpoint
    agent.save_checkpoint(episodes, total_reward)
    env.close()

if __name__ == "__main__":
    main()