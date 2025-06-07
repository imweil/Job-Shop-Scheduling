def sort_two_list_from_big_to_small(sort_list, accord_list):
    combined = list(zip(sort_list, accord_list))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    sorted_sort_list = [item[0] for item in sorted_combined]
    return sorted_sort_list


def sort_two_list_from_small_to_big(sort_list, accord_list):
    combined = list(zip(sort_list, accord_list))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=False)
    sorted_sort_list = [item[0] for item in sorted_combined]
    return sorted_sort_list


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def max_in_columns(matrix):
    return [max(column) for column in zip(*matrix)]


def elementwise_multiply(list1, list2):
    return [a * b for a, b in zip(list1, list2)]


def elementwise_add(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


def elementwise_subtract(list1, list2):
    return [a - b for a, b in zip(list1, list2)]
