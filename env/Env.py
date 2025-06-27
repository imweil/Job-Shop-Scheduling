from src.env.Operation import Operation
from src.env.Job import Job

from src.config.base_config import generate_config, env_config
from src.config.processing_time import processing_time_config

import src.utils.utility as utility


class Env:
    def __init__(self):
        self.time = 0

        self.in_processing_job_last_finish_time = 0

        self.job_list = []
        self.idle_job_list = []

        self.operation_list = []
        self.generate_operation()

        self.finish_job_num = 0

        self.time_window = env_config["time_window"]

    def clean_env(self):
        self.time = 0

        self.in_processing_job_last_finish_time = 0

        self.job_list = []
        self.idle_job_list = []

        self.operation_list = []
        self.generate_operation()

        self.finish_job_num = 0

        self.time_window = env_config["time_window"]
    def generate_job(self, job_list):
        job_category = -1

        job_id = 0

        for i in job_list:
            job_category += 1

            for j in range(i):
                job = Job(job_id, job_category)

                self.job_list.append(job)

                self.idle_job_list.append(job_id)

                job_id += 1

    def time_pass_update(self):
        """update after time passes"""

        self.time += self.time_window

        # update job
        for i in self.job_list:
            job_finish = i.time_pass_update(self.time_window)

            if job_finish:
                self.finish_job_num += 1
                continue

            if not i.in_process and not i.finish:
                if i.id not in self.idle_job_list:
                    self.idle_job_list.append(i.id)

        self.idle_job_list.sort()

        # update operation
        for i in self.operation_list:
            i.time_pass_update(self.time_window)

    def process_update(self, operation_id, machine_id, job_id):

        self.idle_job_list.remove(job_id)

        job_category = self.job_list[job_id].category

        # update
        processing_time = self.operation_list[operation_id].process_update(machine_id, job_category)

        self.job_list[job_id].process_update(machine_id, processing_time)

    """————————————————————————————————————————————————————————————————————————"""

    def generate_operation(self):

        for i in range(env_config["operation_num"]):
            operation = Operation(i)

            operation.generate_machine_config(processing_time_config["operation_" + str(i)])

            self.operation_list.append(operation)

    def get_state(self):
        job_state = self.job_state()
        machine_state = self.machine_state()

        state = utility.flatten_list([job_state, machine_state])

        return state

    def job_state(self):
        state_matrix = [[0] * (self.job_list[-1].category + 1) for _ in range(len(self.operation_list))]

        for job in self.job_list:
            if not job.finish:
                if not job.in_process:
                    state_matrix[job.operation][job.category] += 1


        state_matrix = utility.flatten_list(state_matrix)

        return state_matrix

    def machine_state(self):
        state_process_matrix = []
        state_time_matrix = []
        for operation in self.operation_list:
            process_list, time_list = operation.get_machine_state()
            state_process_matrix.append(process_list)
            state_time_matrix.append(time_list)

        state_process_matrix = utility.flatten_list(state_process_matrix)
        state_time_matrix = utility.flatten_list(state_time_matrix)
        state_matrix = utility.flatten_list([state_process_matrix, state_time_matrix])

        return state_matrix


if __name__ == '__main__':
    env = Env()

    job_list = [3, 2, 4, 1, 2]
    env.generate_job(job_list)
