import src.utils.matching as matching
from src.config.base_config import env_config
import src.utils.utility as utility


class Scheduling:
    def __init__(self):
        self.matching_pattern = 0

    def matching(self, env):
        operation_num = len(env.operation_list)
        operation_idle_job_list = [[]] * operation_num

        for job_id in env.idle_job_list:
            operation_idle_job_list[env.job_list[job_id].operation].append(job_id)

        for o_n in range(operation_num):
            match_list = self.operation_matching(env.operation_list[o_n], operation_idle_job_list[o_n], env.job_list)

            matching.update_after_match(o_n, match_list, env)

            last_time = matching.operation_last_time(env.operation_list[o_n],match_list,env.job_list)

            if last_time + env.time > env.in_processing_job_last_finish_time:
                env.in_processing_job_last_finish_time = last_time +env.time

    def operation_matching(self, operation, idle_job_list, job_list):
        suitable_job_id_list = matching.exist_job(operation, idle_job_list, job_list)

        match_list = []
        # 是否有 机床 工件进行匹配
        if len(operation.idle_machine_list):
            if suitable_job_id_list:
                match_list = self.select_matching_pattern(operation, suitable_job_id_list, job_list)

        return match_list

    """********************************************************************"""

    def select_matching_pattern(self, operation, suitable_job_id_list, job_list):
        # self.matching_pattern = 4 # 改调度模式，可直接注释

        match_list = matching.pattern_match_list(operation, suitable_job_id_list, job_list, self.matching_pattern)

        return match_list


    def matching_last_time(self, operation, suitable_job_id_list, job_list):
        last_time_list = []
        for pattern in range(env_config["pattern_num"]):
            match_list = matching.pattern_match_list(operation, suitable_job_id_list, job_list, pattern)
            longest_time = 0

            for match_pair in match_list:

                machine_id = match_pair[0]
                job_id = match_pair[1]
                job_category = job_list[job_id].category
                processing_time = operation.machine_list[machine_id].job_time_list[job_category]

                if processing_time > longest_time:
                    longest_time = processing_time

            last_time_list.append(longest_time)

        return last_time_list

    def different_pattern_max_last_time(self, operation_list, suitable_job_id_list, job_list):
        last_time_matrix = []
        for operation in operation_list:
            last_time_list = self.matching_last_time(operation, suitable_job_id_list, job_list)
            last_time_matrix.append(last_time_list)
        time_list = utility.max_in_columns(last_time_matrix)
        return time_list


if __name__ == '__main__':
    pass
