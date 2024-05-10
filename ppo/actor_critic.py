import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=True):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim[0]),
            nn.Tanh()
        )

        self.continuous = continuous
        if continuous:
            # Initialize the standard deviation parameter
            self.log_std = nn.Parameter(torch.zeros(action_dim[0]))

    def forward(self, x):
        action_mean = self.actor(x)
        if self.continuous:
            
            std = torch.exp(self.log_std)
            dist = MultivariateNormal(loc=action_mean, scale_tril=torch.diag(std).unsqueeze(0))
        else:
            dist = Categorical(logits=action_mean)

        return dist
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.critic(x)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, continuous = True):
        super(ActorCritic, self).__init__()

        self.actor = Actor(state_dim, action_dim, continuous = True)
        self.critic = Critic(state_dim, action_dim)

        self.continuous = continuous

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        dist = self.actor(state)

        action = dist.sample()
        if self.continuous:
            action = torch.tanh(action)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        dist = self.actor(state)
        state_values = self.critic(state)
        
        if self.continuous:
            action = action
            action_logprobs = dist.log_prob(action).sum(-1, keepdim=True)  # Summing for multi-dimensional actions
        else:
            action_logprobs = dist.log_prob(action)
        
        dist_entropy = dist.entropy()
        
        if not self.continuous:
            dist_entropy = dist_entropy.sum(-1, keepdim=True)  # Summing for multi-dimensional actions
        
        return action_logprobs, state_values, dist_entropy

