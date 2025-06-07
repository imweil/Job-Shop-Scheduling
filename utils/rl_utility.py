import src.utils.utility as utility
from src.config.base_config import env_config
import torch

def reward_time_window(last_time, now_time, update_last_time, action_probability):
    probability_time = utility.elementwise_multiply(update_last_time, action_probability)
    time = sum(probability_time) / len(probability_time) + now_time
    time = time.item()

    reward = -(last_time - time) / env_config['reward_time_windows_normalization']
    return reward


def reward_pattern(last_time, update_last_time):
    reward = (last_time - update_last_time) / env_config['reward_pattern_normalization']
    return reward


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
