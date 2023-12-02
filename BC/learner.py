import torch
from torch import optim
import numpy as np
from dataset import ExpertData
from torch import distributions as pyd
import torch.nn as nn
import os


# Discrete policy model
class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, states):
        '''Returns action distribution for all states s in our batch.
        
        :param states: torch.Tensor, size (B, state_dim)
        
        :return logits: action logits, size (B, action_dim)
        '''
        logits = self.net(states)
        return logits.float()


# Behavioral cloning (BC) class
class BC:
    def __init__(self, state_dim, action_dim, args):
        # Policy network setup
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.loss = nn.CrossEntropyLoss()

    def get_logits(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        return self.policy(states)

    def learn(self, expert_states, expert_actions):
        # Get logits under our policy
        logits = self.get_logits(expert_states)
        # Compute loss with target policy
        loss = self.loss(logits, expert_actions)
        # Backward step
        self.optimizer.zero_grad()
        loss.backward()
        # Update the policy
        self.optimizer.step()
        # Return current loss for saving
        return loss.item()

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
