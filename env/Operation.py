import random

from src.env.Machine import Machine

from src.config.base_config import generate_config, env_config

import src.utils.utility as utility


class Operation:
    def __init__(self, operation_id):
        self.id = operation_id
        self.machine_list = []
        self.idle_machine_list = []

        self.time_normalization_factor = env_config["time_normalization_factor"]

    def generate_machine_config(self, machine_dic):
        for i in range(len(machine_dic)):
            time_list = machine_dic["machine" + str(i)]
            machine = Machine(i, self.id, time_list)
            self.machine_list.append(machine)

            self.idle_machine_list.append(i)

    def time_pass_update(self, time_window):
        self.idle_machine_list = []

        for i in self.machine_list:
            i.time_pass_update(time_window)
            if not i.in_process:
                self.idle_machine_list.append(i.id)

    def process_update(self, machine_id, job_category):

        processing_time = self.machine_list[machine_id].process_update(job_category)

        self.idle_machine_list.remove(machine_id)

        return processing_time

    def return_idle_wait_time(self):
        idle_machine_num = len(self.idle_machine_list)
        wait_time_list = [0] * idle_machine_num

        for i in range(idle_machine_num):
            wait_time_list[i] = self.machine_list[self.idle_machine_list[i]].waiting_time

        return wait_time_list

    def sort_machine_wait_form_big_to_small(self):
        """给空闲的机器的等待时间排序，重大到小"""
        wait_time_list = self.return_idle_wait_time()

        sorted_machine_id_list = utility.sort_two_list_from_big_to_small(self.idle_machine_list, wait_time_list)
        return sorted_machine_id_list

    def get_machine_state(self):
        state_list = [0] * len(self.machine_list)
        time_list = state_list[:]

        for machine in self.machine_list:
            if not machine.in_process:
                state_list[machine.id] = 1
            else:
                time_list[machine.id] = machine.process_time / self.time_normalization_factor

        return state_list, time_list


"""示例"""
if __name__ == '__main__':
    a = Operation(1)
    a.generate_machine_random(2)
    print(a.machine_list[1].job_1)
