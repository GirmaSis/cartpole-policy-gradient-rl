import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
import imageio  # For saving GIFs
import os
import cv2 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym

os.makedirs("gifs_REINFORCE", exist_ok=True)

# Argument parser to set hyperparameters
parser = argparse.ArgumentParser(description='Optimized PyTorch REINFORCE with Baseline for CartPole')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=543, help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, help='interval between training status logs')
args = parser.parse_args()

# Initialize environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    """
    Implements REINFORCE with a baseline (value estimation)
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.dropout(x)
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    R = 0
    policy_losses = []
    value_losses = []
    returns = []
    
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for (log_prob, value), R in zip(model.saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

def modify_pole_color(frame):
    """ Changes the color of the pole to red """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    frame[mask > 0] = [0, 0, 255]  # Change black pole to red
    return frame

def save_gif(frames, episode):
    """ Saves the episode as a GIF """
    imageio.mimsave(f'gifs_REINFORCE/cartpole_episode_{episode}.gif', frames, fps=30)

def main():
    running_reward = 10
    rewards_log = []
    avg_rewards = []
    
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        frames = []

        for t in range(1, 500):
            frame = env.render()
            frame = modify_pole_color(frame)  # Change pole color
            frames.append(frame)
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        rewards_log.append(ep_reward)
        avg_rewards.append(running_reward)
        finish_episode()
        
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
            save_gif(frames, i_episode)  # Save GIF every log interval
        
        if i_episode > 30 and np.mean(rewards_log[-30:]) > 180:
            print("Training complete! Model has converged.")
            break

    # Plot training performance
    plt.plot(rewards_log, label='Episode Reward', color='blue')
    plt.plot(avg_rewards, label='Running Avg Reward', linestyle='dashed', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Performance: Optimized REINFORCE with Baseline')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
