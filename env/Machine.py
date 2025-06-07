import src.utils.utility as utility


class Machine:
    def __init__(self, machine_id, operation, job_processing_time):
        self.id = machine_id
        self.operation = operation

        self.job_time_list = job_processing_time

        for i in range(len(job_processing_time)):
            setattr(self, f'job_{i}', job_processing_time[i])


        self.in_process = False

        self.process_time = 0

        self.waiting_time = 0

        self.job_category_big_list = self.sort_job_category_big_list()

        self.job_category_small_list = self.job_category_big_list[::-1]

    def time_pass_update(self, time_window):

        if self.in_process:
            self.process_time -= time_window

            if self.process_time <= 0:
                self.in_process = False
                self.waiting_time = - self.process_time
                self.process_time = 0

        else:
            self.waiting_time += time_window

    def process_update(self, job_category):
        processing_time = getattr(self, f'job_{job_category}')
        self.in_process = True
        self.process_time = processing_time
        self.waiting_time = 0
        return processing_time

    def sort_job_category_big_list(self):
        job_time_list = self.job_time_list[:]
        category_num = len(job_time_list)
        category_list = list(range(category_num))
        return utility.sort_two_list_from_big_to_small(category_list, job_time_list)
