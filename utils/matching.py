from src.config.base_config import env_config
import src.utils.utility as utility
import numpy as np


def exist_job(operation, idle_job_list, job_list):
    """是否存在可匹配工件，返回id_list"""

    operation_id = operation.id
    suitable_job_id = []

    for i in idle_job_list:
        if job_list[i].operation == operation_id:
            suitable_job_id.append(i)

    return suitable_job_id


def update_after_match(operation_id, match_list, env):
    """完成匹配后更新"""

    for match_pair in match_list:
        env.process_update(operation_id, match_pair[0], match_pair[1])


def id_list_turn_category_num(job_id_list, job_list):
    """给一个工件的id的序列，输出各种工件的类型的数量"""

    category_list = [0] * env_config["job_category_num"]

    for job_id in job_id_list:
        category = job_list[job_id].category

        category_list[category] += 1

    return category_list


def sort_idle_time_job_from_big_to_small(job_id_list, job_list):
    wait_time_list = []
    copy_id_list = job_id_list[:]

    for id in job_id_list:
        wait_time = job_list[id].waiting_time
        wait_time_list.append(wait_time)

    copy_id_list = utility.sort_two_list_from_big_to_small(copy_id_list, wait_time_list)

    return copy_id_list


def match_long_job(machine, id_list, job_list):
    for j in machine.job_category_big_list:

        for i in id_list:
            if job_list[i].category == j:
                id_list.remove(i)

                return i


def match_short_job(machine, id_list, job_list):

    for j in machine.job_category_small_list:

        for i in id_list:
            if job_list[i].category == j:
                id_list.remove(i)

                return i


# def match_long_wait_job(machine, id_list, job_list):
#     for j in machine.job_category_small_list:
#
#         for i in id_list:
#             if job_list[i].category == j:
#                 id_list.remove(i)
#
#                 return i


"""**************************************匹配方法**************************************"""


def machine_wait_long_job_process_long(operation, suitable_job_id_list, job_list):
    """机器等待时间最长先选择最长加工时间的"""

    sorted_machine_id_list = operation.sort_machine_wait_form_big_to_small()

    copy_job_list = suitable_job_id_list[:]

    match_list = []

    for machine_id in sorted_machine_id_list:
        if copy_job_list:
            match_job_id = match_long_job(operation.machine_list[machine_id], copy_job_list, job_list)
            match_list.append([machine_id, match_job_id])

        else:
            break

    return match_list


def machine_wait_long_job_process_short(operation, suitable_job_id_list, job_list):
    """机器等待时间最长先选择最短加工时间的"""

    sorted_machine_id_list = operation.sort_machine_wait_form_big_to_small()

    copy_job_list = suitable_job_id_list[:]

    match_list = []

    for machine_id in sorted_machine_id_list:
        if copy_job_list:
            match_job_id = match_short_job(operation.machine_list[machine_id], copy_job_list, job_list)
            match_list.append([machine_id, match_job_id])

        else:
            break

    return match_list


def machine_wait_long_job_wait_long(operation, suitable_job_id_list, job_list):
    """机器等待时间最长先选择等待时间长的工件"""

    sorted_machine_id_list = operation.sort_machine_wait_form_big_to_small()

    sort_job_list = sort_idle_time_job_from_big_to_small(suitable_job_id_list, job_list)

    match_list = []

    list_len = min(len(sorted_machine_id_list), len(sort_job_list))

    for i in range(list_len):
        match_list.append([sorted_machine_id_list[i], sort_job_list[i]])

    return match_list


def short_process_time_first(operation, suitable_job_id_list, job_list):
    """最短加工时间"""
    category_list = id_list_turn_category_num(suitable_job_id_list, job_list)
    matrix, operation.idle_machine_list = idle_machine_time_matrix(operation)
    result = greedy_shortest_match(matrix, category_list)

    match_list = job_category_find_machine(operation, category_list, suitable_job_id_list, result)

    return match_list


def long_process_time_first(operation, suitable_job_id_list, job_list):
    """最长加工时间"""
    category_list = id_list_turn_category_num(suitable_job_id_list, job_list)
    matrix, operation.idle_machine_list = idle_machine_time_matrix(operation)
    result = greedy_longest_match(matrix, category_list)

    match_list = job_category_find_machine(operation, category_list, suitable_job_id_list, result)

    return match_list


def idle_machine_time_matrix(operation):
    matrix = []
    for i in operation.idle_machine_list:
        matrix.append(operation.machine_list[i].job_time_list)

    return matrix, operation.idle_machine_list


def job_category_find_machine(operation, category_num_list, suitable_job_id_list, result):
    """根据工件种类，匹配工件和机器"""
    matched_list = [0] * len(category_num_list)
    match_list = []
    for i in result:
        machine_id = operation.idle_machine_list[i[0]]
        job_category = i[1]
        job_list_id = matched_list[job_category]
        for j in range(job_category):
            job_list_id += category_num_list[j]

        job_id = suitable_job_id_list[job_list_id]

        match_list.append([machine_id, job_id])

        matched_list[job_category] += 1

    return match_list


def greedy_shortest_match(processing_times, workpieces):
    num_machines = len(processing_times)
    num_workpieces = len(workpieces)

    # 创建一个可匹配的列表
    matches = []

    # 记录每种工件的剩余数量
    available_workpieces = workpieces.copy()

    # 逐步选择每台机器的匹配
    for i in range(num_machines):
        min_time = float('inf')
        chosen_workpiece = -1

        # 寻找当前机器最小加工时间的可用工件
        for j in range(num_workpieces):
            if available_workpieces[j] > 0 and processing_times[i][j] < min_time:
                min_time = processing_times[i][j]
                chosen_workpiece = j

        if chosen_workpiece != -1:  # 如果找到了可用的工件
            matches.append((i, chosen_workpiece))  # 记录匹配
            available_workpieces[chosen_workpiece] -= 1  # 减少工件数量

    return matches


def greedy_longest_match(processing_times, workpieces):
    num_machines = len(processing_times)
    num_workpieces = len(workpieces)

    # 创建一个可匹配的列表
    matches = []

    # 记录每种工件的剩余数量
    available_workpieces = workpieces.copy()

    # 逐步选择每台机器的匹配
    for i in range(num_machines):
        max_time = -1
        chosen_workpiece = -1

        # 寻找当前机器最长加工时间的可用工件
        for j in range(num_workpieces):
            if available_workpieces[j] > 0 and processing_times[i][j] > max_time:
                max_time = processing_times[i][j]
                chosen_workpiece = j

        if chosen_workpiece != -1:  # 如果找到了可用的工件
            matches.append((i, chosen_workpiece))  # 记录匹配
            available_workpieces[chosen_workpiece] -= 1  # 减少工件数量

    return matches


def pattern_match_list(operation, suitable_job_id_list, job_list, pattern):
    match_list = []
    if pattern == 0:
        match_list = machine_wait_long_job_process_long(operation, suitable_job_id_list, job_list)
    if pattern == 1:
        match_list = machine_wait_long_job_process_short(operation, suitable_job_id_list, job_list)
    if pattern == 2:
        match_list = machine_wait_long_job_wait_long(operation, suitable_job_id_list, job_list)
    if pattern == 3:
        match_list = short_process_time_first(operation, suitable_job_id_list, job_list)
    if pattern == 4:
        match_list = long_process_time_first(operation, suitable_job_id_list, job_list)

    return match_list


def operation_last_time(operation, match_list,job_list):
    time = 0
    for match_pair in match_list:
        machine_id = match_pair[0]
        job_id = match_pair[1]
        job_category = job_list[job_id].category
        processing_time = operation.machine_list[machine_id].job_time_list[job_category]

        if processing_time > time:
            time = processing_time

    return time
