from src.agent.agent import Agent
from src.env.Env import Env
from src.env.Scheduling import Scheduling
import src.utils.rl_utility as rl_utility
import random
import time

start_time = time.time()

for a in range(5,6):
    agent = Agent()
    agent.load("110000")
    env = Env()
    schedule_center = Scheduling()

    job_list = [random.randint(1, 15) for _ in range(7)]
    job_list = [4, 2, 4, 2, 3, 3, 2]
    for i in range(len(job_list)):
        job_list[i] = job_list[i] * a

    env.generate_job(job_list)
    nest_state = env.get_state()

    while env.finish_job_num != len(env.job_list):
        state = nest_state

        window, scheduling_pattern, window_probs, scheduling_pattern_probs, embedding = agent.take_action(state)
        schedule_center.matching_pattern = scheduling_pattern

        env.time_window = window + 5

        env.time_window = 25 # 无自适应时间窗可注释本行
        last_time = env.in_processing_job_last_finish_time
        update_last_time = schedule_center.different_pattern_max_last_time(env.operation_list, env.idle_job_list,
                                                                           env.job_list)

        schedule_center.matching(env)
        nest_state = env.get_state()
        env.time_pass_update()

    print(env.time)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"运行时间为 {elapsed_time:.6f} 秒")