import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gymnasium as gym
from tqdm import trange
import statistics


from ppo.utils import RolloutBuffer
from ppo.actor_critic import ActorCritic
from refinement.goal import Goal
# from refinement.graph import Node
from refinement.utils import CacheStates
import pickle

class PPO:
    def __init__(self, state_dim, action_dim, continuous, lr_actor, lr_critic, gamma, K_epochs, eps_clip, batch_size = 256, device = "cpu"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __call__(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.cpu().numpy()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = rewards.reshape(-1, 1)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        _, old_state_values, _ = self.policy.evaluate(old_states, old_actions)
        advantages = rewards - old_state_values.detach()
        # calculate advantages
        # Create data loader for batch processing
        dataset = TensorDataset(old_states, old_actions, old_logprobs, rewards, advantages)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for states_, actions_, logprobs_, reward, advantage in data_loader:
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(states_, actions_)

                # Match state_values tensor dimensions with rewards tensor
                # state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - logprobs_.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                # Final loss of clipped objective PPO
                # print(state_values)
                loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, reward) - 0.01 * dist_entropy

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)  # Gradient clipping
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))


def sample_policy(env: gym.Env, observation:np.ndarray, policy:PPO, goal:Goal):
    
    final_terminated = False
    total_reward = 0
    traj = [observation]
    while True:
        # print(observation)
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        traj.append(observation)
        # env.render()
        total_reward+=reward

        policy.buffer.rewards.append(reward)

        # if goal.predicate(observation):
        final_terminated = info['is_success']
        
        policy.buffer.is_terminals.append(terminated)

        if final_terminated or terminated or truncated:
            break
    
    return final_terminated, total_reward, info, traj

def train_policy(env: gym.Env, start_node, end_node, n_episodes=3000, minimum_reach=0.9):


    env.set_abstract_states(start_node, end_node)
    print(env.observation_space.shape)
    policy = PPO(
        state_dim = env.observation_space.shape,
        action_dim = env.action_space.shape,
        continuous=True,
        lr_actor=0.0001,
        lr_critic=0.0001,
        gamma = 0.99,
        K_epochs = 10,
        eps_clip = 0.1,
        device = "cpu"
    )

    reach = []
    rewards = [0]
    episodes = trange(n_episodes, desc='reach')
    len_traj = [0]
    for episode in episodes:
        
        end_node.goal.reset()
        observation, _ = env.reset()
        # env.render()
        
        reached, reward, info, traj= sample_policy(env, observation, policy, end_node.goal)
        reach.append(reached)
        rewards.append(reward)
        len_traj.append(len(traj))

        episodes.set_description(f"Current reach: {sum(reach)/len(reach):.2f}, total_reach: {sum(reach)}, reward: {statistics.mean(rewards):.2f}±{statistics.stdev(rewards):.1f}, ep_len: {statistics.mean(len_traj):.2f}±{statistics.stdev(len_traj):.1f}")
        
        if sum(reach[-1000:])/(1000) > minimum_reach:
            break

        if (episode+1) % 100 == 0:
            policy.update()
        
    print(policy.policy.actor.log_std)
            
    return policy


def test_policy(policy: PPO, env: gym.Env, start_node, end_node, n_episodes=3000):
    
    env.set_abstract_states(start_node, end_node)
    
    cached_states = CacheStates()

    reach = []
    rewards = [0]
    episodes = trange(n_episodes, desc='reach')
    trajectories = []
    for episode in episodes:
        
        # start_state = start_node.sample_state()
        goal_observation = end_node.goal.reset()
        observation, _ = env.reset()
        # env.render()
        
        reached, reward, info, traj= sample_policy(env, observation, policy, end_node.goal)
        reach.append(reached)
        rewards.append(reward)
        trajectories.append((traj, reached))
        cached_states.insert(goal_observation, reached)

        episodes.set_description(f"Current reach: {sum(reach)/len(reach):.2f}, total_reach: {sum(reach)}, reward: {statistics.mean(rewards):.2f}±{statistics.stdev(rewards):.1f}")
        
    pickle.dump(trajectories, open(f"{start_node.name}_{end_node.name}_traj.pkl", "wb"))    
    pickle.dump(cached_states.return_dataset(), open(f"{end_node.name}_data.pkl", "wb"))

    return sum(reach)/len(reach), cached_states
    
