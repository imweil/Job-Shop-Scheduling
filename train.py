from src.agent.agent import Agent
from src.env.Env import Env
from src.env.Scheduling import Scheduling
import src.utils.rl_utility as rl_utility
import random
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

agent = Agent()

env = Env()
schedule_center = Scheduling()

for i in tqdm(range(10 ** 8), desc="Training Progress", unit="iteration"):
    env.clean_env()

    job_list = [random.randint(1, 15) for _ in range(7)]

    env.generate_job(job_list)

    next_state = env.get_state()

    transition_time_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }

    transition_pattern_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    done = False
    while not done:
        state = next_state

        window, scheduling_pattern, window_probs, scheduling_pattern_probs, embedding = agent.take_action(state)
        schedule_center.matching_pattern = scheduling_pattern
        env.time_window = window + 5
        last_time = env.in_processing_job_last_finish_time

        update_last_time = schedule_center.different_pattern_max_last_time(env.operation_list, env.idle_job_list,
                                                                           env.job_list)

        time_reward = rl_utility.reward_time_window(last_time, env.time, update_last_time, window_probs)

        schedule_center.matching(env)
        pattern_reward = rl_utility.reward_pattern(last_time, env.in_processing_job_last_finish_time)

        next_state = env.get_state()
        env.time_pass_update()

        if env.finish_job_num == len(env.job_list):
            done = True
            if transition_pattern_dict['next_states']:
                transition_pattern_dict['next_states'].pop(0)
                transition_pattern_dict['next_states'].append(embedding.tolist())

        transition_time_dict['states'].append(state)
        transition_time_dict['actions'].append(window)
        transition_time_dict['next_states'].append(next_state)
        transition_time_dict['rewards'].append(time_reward)
        transition_time_dict['dones'].append(done)

        transition_pattern_dict['states'].append(embedding.tolist())
        transition_pattern_dict['actions'].append(scheduling_pattern)
        transition_pattern_dict['next_states'].append(embedding.tolist())
        transition_pattern_dict['rewards'].append(pattern_reward)
        transition_pattern_dict['dones'].append(done)

    agent.update_windows(transition_time_dict)
    agent.update_pattern(transition_pattern_dict)

    if i % 1 == 0:
        agent.save(str(i))
