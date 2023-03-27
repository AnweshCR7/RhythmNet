import glob
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
import torch

TARGETS = {
    'SYS': 'Systolic',
    'DIA': 'Diastolic',
    'GLUCOSE': 'Sugar Level',
    'HR': 'Heart Rate (bpm)',
    'SpO2': 'Oxygen Saturation (%)'
}


def is_gpu_available():
    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(42)
    else:
        print("GPU not available, using CPU")


def get_dataframe(csv_dir, dataset_name: str):
    folder = os.path.join(csv_dir, dataset_name)
    paths = glob.glob(f'{folder}/*.csv')
    dataframes = [pd.read_csv(path) for path in paths]
    return dataframes


def rolling_window(dataframe, window_size, step_size):
    """Apply rolling window for each csv separately"""
    splits = dataframe.rolling(window_size, step=step_size)

    splits = list(splits)[math.ceil(window_size / step_size):]
    return splits


def get_target_col(json_path: str) -> str:
    with open(json_path, "r") as _json:
        json_content = json.load(_json)

    target = json_content['target']
    target_col = TARGETS[target]

    return target_col


def convert_to_timestamp_obj(data_frame):
    """Converting the time column to a Timestamp object

    THis enables us to utilize the rich set of time-series manipulation
    functions provided by the Pandas library
    """

    data_frame['time'] = data_frame['time'].apply(replace_colon_with_dot)
    data_frame['time'] = pd.to_datetime(data_frame['time'], format='%H:%M:%S.%f')
    data_frame.set_index('time', inplace=True)
    return data_frame


def replace_colon_with_dot(time_str: str):
    return time_str[:8] + '.' + time_str[9:]


def format_time_col(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    formatted_dataframes = []
    for data_frame in dataframes:
        # remove an empty space from Date & Time (Formate : dd MMM yyyy HH:mm:ss:SSS Z) column
        data_frame['time'] = data_frame['Date & Time (Formate : dd MMM yyyy HH:mm:ss:SSS Z)'].apply(lambda x: x.strip())
        data_frame = data_frame.drop('Date & Time (Formate : dd MMM yyyy HH:mm:ss:SSS Z)', axis=1)

        # from time extract Hour,Min,Sec information
        data_frame['time'] = data_frame['time'].apply(lambda x: x.split(' ')[3])
        data_frame = convert_to_timestamp_obj(data_frame)
        formatted_dataframes.append(data_frame)

    return formatted_dataframes


def resample_df(data_frame: pd.DataFrame, milisec: int):
    """Resamples the input DataFrame to a specified frequency in milliseconds.

    Notes:
        This function uses the `first()` method to return the first value of each
        resampled interval.The `first()` method returns the first value in each group
        of values.In the context of resampling,this means that the first value of each
        resampled interval is returned.
    """

    return data_frame.resample(f'{milisec}L').first()


def select_yuv_columns(data_frame: pd.DataFrame) -> np.ndarray:
    columns_name = list(data_frame.columns)
    features_col = [col for col in columns_name if col.startswith('Y') or col.startswith('U') or col.startswith('V')]
    data_frame = data_frame[features_col]
    return data_frame.values


def get_st_maps(dataframes, window_size, step_size):
    sp_maps = [rolling_window(data_frame, window_size, step_size) for data_frame in dataframes]

    # convert a nested list into a flat list
    return list(itertools.chain(*sp_maps))
