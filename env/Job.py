from src.config.base_config import env_config


class Job:
    def __init__(self, job_id, job_category):
        self.id = job_id

        self.category = job_category

        self.finish = False

        self.in_process = False

        self.machine = None

        self.processing_remain_time = 0

        self.operation = 0

        self.waiting_time = 0

        self.operation = 0

    def time_pass_update(self, time_window):
        if not self.finish:

            if self.in_process:
                self.processing_remain_time -= time_window

                if self.processing_remain_time <= 0:
                    self.in_process = False
                    self.waiting_time = -self.processing_remain_time
                    self.processing_remain_time = 0

                    self.operation += 1

                    if self.operation >= env_config["operation_num"]:
                        self.finish = True
                        return True

            else:
                self.waiting_time += time_window

        return False

    def process_update(self, machine_id, processing_remain_time):
        self.in_process = True
        self.machine = machine_id
        self.processing_remain_time = processing_remain_time
        self.waiting_time = 0
