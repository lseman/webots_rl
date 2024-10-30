import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from datetime import timedelta
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    """Neural network for the policy (actor) and value (critic) functions."""

    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        # Actor network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.actor = nn.Linear(hidden_size * 2, output_size)
        
        # Critic network layers
        self.critic_fc1 = nn.Linear(input_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.critic = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        """Compute action probabilities and state value."""
        # Actor forward pass
        x_actor = torch.relu(self.fc1(x))
        x_actor = torch.relu(self.fc2(x_actor))
        action_probs = torch.softmax(self.actor(x_actor), dim=-1)
        
        # Critic forward pass
        x_critic = torch.relu(self.critic_fc1(x))
        x_critic = torch.relu(self.critic_fc2(x_critic))
        state_value = self.critic(x_critic)
        
        return action_probs, state_value


class PPOAgent:
    """Agent implementing the PPO algorithm."""

    def __init__(self, input_size, hidden_size, output_size, num_episodes, max_steps, learning_rate, gamma, clip_ratio, update_epochs):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        
        self.network = PolicyNetwork(input_size, hidden_size, output_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), self.learning_rate)
        
        self.env = Environment()  # Assume environment is defined elsewhere
    
    def compute_advantages(self, rewards, values):
        """Compute advantages using GAE (Generalized Advantage Estimation)."""
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        return torch.tensor(advantages, dtype=torch.float32, device=device), torch.tensor(returns, dtype=torch.float32, device=device)
    
    def ppo_update(self, log_probs, states, actions, returns, advantages):
        for _ in range(self.update_epochs):
            new_log_probs, new_values = [], []
            for state, action in zip(states, actions):
                action_probs, value = self.network(state)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs.append(dist.log_prob(action))
                new_values.append(value)
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()
            
            # Calculate the ratios
            ratios = torch.exp(new_log_probs - log_probs)
            
            # Surrogate objective function
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.functional.mse_loss(new_values, returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        reward_history = []
        best_score = -np.inf
        for episode in range(self.num_episodes):
            state = torch.as_tensor(self.env.reset(), dtype=torch.float32, device=device)
            
            log_probs = []
            states = []
            actions = []
            rewards = []
            values = []
            done = False
            ep_reward = 0
            
            for _ in range(self.max_steps):
                action_probs, state_value = self.network(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                next_state, reward, done = self.env.step(action.item())
                ep_reward += reward
                
                # Store experience
                log_probs.append(log_prob)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(state_value)
                
                state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
                
                if done:
                    break
            
            # Store final state value (0 if done, else value of last state)
            values.append(torch.tensor([0.0], device=device) if done else self.network(state)[1])
            
            # Compute returns and advantages
            advantages, returns = self.compute_advantages(rewards, values)
            
            # Update policy and value function
            self.ppo_update(torch.stack(log_probs), states, actions, returns, advantages)
            reward_history.append(ep_reward)
            
            # Save model if current reward is the best so far
            if ep_reward > best_score:
                self.save('/best_weights.pt')
                best_score = ep_reward
            
            print(f"Episode {episode + 1}: Score = {ep_reward:.3f}")
        
        self.save('/final_weights.pt')
        self.plot_rewards(reward_history)
    
    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def plot_rewards(self, rewards):
        sma = np.convolve(rewards, np.ones(25) / 25, mode='valid')
        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.show()
