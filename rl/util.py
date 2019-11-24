import os, numpy as np, torch as th

def compute_returns(vprime, rewards, masks, gamma=0.99):
    R,returns = vprime.detach(),[]
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * masks[t]
        returns.insert(0, R)
    return returns

def normalize_rewards(rewards, gamma=0.99):
    rewards = torch.tensor(discount_rewards(rewards, gamma))
    denom = rewards.std() + np.finfo(np.float32).eps.item()
    rewards = (rewards - rewards.mean())/(denom)
    return rewards

def estimate_advantage(vprime, values, rewards, masks, gamma=0.99, tau=0.95):
    values = th.cat((values,vprime),dim=0).detach()
    gae,returns = 0,[]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        gae = delta + gamma * tau * masks[t] * gae
        returns.insert(0, gae + values[t])
    return returns

def mask_actions(legal_actions, action_space):
    mask = th.zeros((1, action_space))
    mask[:,legal_actions] = 1
    return mask
