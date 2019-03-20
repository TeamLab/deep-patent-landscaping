import pandas as pd
import os
from itertools import permutations
import numpy as np


def get_dup_label(data, series):
    valid_set = set()
    garbage_set = set()
    for codes in series.loc[data.valid == True]:
        for code in codes:
            valid_set.add(code)
    for codes in series.loc[data.valid == False]:
        for code in codes:
            garbage_set.add(code)

    duplicated = {valid_code for valid_code in valid_set if valid_code in garbage_set}

    print("valid :", len(valid_set), "garbage :", len(garbage_set), "duplicated :", len(duplicated))
    return list(duplicated), list(valid_set), list(garbage_set)


def series_convert_to_list(data):
    data.uspc_code = data.uspc_code.apply(lambda x: list(set(x.split(','))))
    data.ipc_code = data.ipc_code.apply(lambda x: list(set(x.split(','))))
    data.cpc_code = data.cpc_code.apply(lambda x: list(set(x.split(','))))
    data.refs = data.refs.apply(lambda x: list(set(x.split(','))))


def remove_duplicated_code(series, duplicated):
    column = []
    for codes in series:
        unique = []
        for code in codes:
            if code not in duplicated:
                unique.append(code)
        column.append(unique)

    return column
