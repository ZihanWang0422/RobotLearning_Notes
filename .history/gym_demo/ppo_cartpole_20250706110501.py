import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# 超参数
gamma = 0.99
clip_epsilon = 0.2
lr = 3e-4
update_steps = 5
batch_size = 2048
epochs = 1000

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logprobs, torch.squeeze(values), entropy

def compute_gae(rewards, masks, values, next_value, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(model, optimizer, states, actions, logprobs, returns, advantages):
    for _ in range(update_steps):
        new_logprobs, state_values, entropy = model.evaluate(states, actions)
        ratio = (new_logprobs - logprobs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(state_values, returns)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        state = env.reset()
        states, actions, rewards, logprobs, masks, values = [], [], [], [], [], []
        ep_reward = 0
        for _ in range(batch_size):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, logprob = model.act(state_tensor)
            value = model.forward(state_tensor)[1]
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            logprobs.append(logprob.item())
            masks.append(1 - done)
            values.append(value.item())
            state = next_state
            ep_reward += reward
            if done:
                state = env.reset()
        next_state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_value = model.forward(next_state_tensor)[1].item()
        returns = compute_gae(rewards, masks, values, next_value)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        logprobs = torch.FloatTensor(logprobs)
        returns = torch.FloatTensor(returns)
        advantages = returns - torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ppo_update(model, optimizer, states, actions, logprobs, returns, advantages)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Reward: {ep_reward/batch_size:.2f}")

if __name__ == "__main__":
    main()
