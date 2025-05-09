import numpy as np
import pickle
import os
from os.path import join
import pandas as pd
import multiprocessing
from tqdm import tqdm
import math
from datetime import datetime, timedelta, timezone
from functools import partial
from loader.preprocess.mm.utils import gcj02_to_wgs84


# 用于进程间共享的统计变量
invalid_timestamps = multiprocessing.Value('i', 0)  # 丢弃的无效时间戳数据点
over_time_range = multiprocessing.Value('i', 0)  # 丢弃的时间小于五分钟或者大于两小时或者跨天的轨迹数据点
total_dis_traj = multiprocessing.Value('i', 0)  # 总的丢弃的轨迹数据点
empty_value = multiprocessing.Value('i', 0)  # 丢弃的空值数据点


def convert_to_trajectory(group):
    global invalid_timestamps,total_dis_traj
    trajectory_id = group['OrderId'].iloc[0]  # 通过 group 提取轨迹编号
    trajectory = []
    for time, lng, lat in group[['Timestamp', 'Lng', 'Lat']].values:
        # 检查时间戳是否有效
        if pd.isna(time):  # 如果时间戳无效
            invalid_timestamps.value += 1  # 无效时间戳数据点计数
            total_dis_traj.value += 1
            continue
        # lng, lat = gcj02_to_wgs84(lng, lat)
        trajectory.append((int(time), lat, lng, trajectory_id))  # 加入轨迹编号，四元组代表时间戳、纬度、经度、轨迹编号
    return trajectory


def convert_single(group, time_zone):
    global empty_value, total_dis_traj, over_time_range

    group = group.sort_values(by='Timestamp').reset_index()
    group = group.drop(columns="index", axis=1, inplace=False)

    # 确保 'Timestamp' 列是整数类型
    group['Timestamp'] = pd.to_numeric(group['Timestamp'], errors='coerce')  # 将 Timestamp 转换为数值类型

    # 处理 NaN 值：删除包含 NaN 的行
    group = group.dropna(subset=['Timestamp'])

    # 检查转换后是否有空值
    if group.empty:
        empty_value.value += len(group)  # 空值数据点计数
        total_dis_traj.value += len(group)
        return None  # 如果是空值，则返回 None

    beg, end = group.index[0], group.index[-1]
    duration = group.at[end, 'Timestamp'] - group.at[beg, 'Timestamp']

    if duration <= 300 or duration > 7200:
        over_time_range.value += len(group)  # 丢弃轨迹
        total_dis_traj.value += len(group)
        return None

    init_timestamp = int(group.at[beg, 'Timestamp'])
    finish_timestamp = int(group.at[end, 'Timestamp'])
    init_dt = datetime.fromtimestamp(init_timestamp, time_zone)
    finish_dt = datetime.fromtimestamp(finish_timestamp, time_zone)

    if init_dt.day != finish_dt.day:
        over_time_range.value += len(group)  # 丢弃跨天轨迹
        total_dis_traj.value += len(group)
        return None

    return convert_to_trajectory(group)


def get_trajectories(date, raw_traj_path):
    print(f"processing date {date}...")

    # 为数据文件指定明确的列名
    column_names = ['OrderId', 'CarId', 'Timestamp', 'Lng', 'Lat']

    # 读取 CSV 文件，并指定每一列的数据类型
    dtype = {
        'OrderId': str,  # 轨迹编号为字符串类型
        'CarId': str,  # CarId 列为字符串类型，可以删除
        'Timestamp': int,  # 时间戳为整数类型
        'Lng': float,  # 经度为浮动数值类型
        'Lat': float  # 纬度为浮动数值类型
    }

    # 读取 CSV 文件并为其指定列名及数据类型

    # 筛选8~16点的数据，其他的数据不要
    data = pd.read_csv(open(join(raw_traj_path, f"xian_first_preprocessed_{date}_shrink.csv"), "r"), header=0, names=column_names,
                       dtype=dtype)
    print(f"当前读取的文件为xian_first_preprocessed_{date}_shrink.csv")

    data = data.drop(columns=['CarId'], axis=1)  # 删除 CarId 列
    print("读取完成!")
    print(f"初始文件一共有{len(data)}条数据")
    print("初始数据样例:", data.head())

    # # # 筛选出当天 8 点到 16 点的数据
    # data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')  # 将时间戳转换为datetime类型
    # data = data[data['Timestamp'].dt.hour >= 8]  # 8点之后的数据
    # data = data[data['Timestamp'].dt.hour < 16]  # 16点之前的数据
    #
    # # 获取筛选后的数据的时间范围
    # start_time = data['Timestamp'].min() # 最早的时间
    # end_time = data['Timestamp'].max() # 最晚的时间
    #
    # print(f"筛选后的数据有 {len(data)} 条")
    # print(f"筛选后的时间范围是：{start_time} 到 {end_time}")
    #
    # # 转回原始时间戳格式（单位：秒）
    # data['Timestamp'] = data['Timestamp'].astype('int64') // 10 ** 9  # 将datetime转换为秒级的时间戳
    # # # 打印一下转换后的数据
    # # print("筛选后数据样例:", data.head())

    traj_grouped = data.groupby(by=['OrderId'], axis=0)  # 按照 OrderId 分组
    print(f"分组后一共有{len(traj_grouped)}条轨迹")
    n_process = min(int(os.cpu_count()) + 1, 30)

    trajectories = []
    time_zone = timezone(timedelta(hours=8))  # 设置时区为东八区
    partialprocessParallel = partial(convert_single, time_zone=time_zone)
    with multiprocessing.Pool(n_process) as pool:
        results = list(
            tqdm(pool.imap(partialprocessParallel, [group for _, group in traj_grouped]), total=len(traj_grouped),
                 ncols=80))

    trajectories = [each for each in results if each is not None]

    # 计算总的数据条数（每个轨迹包含多个点）
    total_data_points = sum(len(trajectory) for trajectory in trajectories)

    # 打印总数据条数
    print(f"文件经过初步处理后一共还有{total_data_points}条数据")
    print(f"总共丢弃了 {total_dis_traj.value} 条轨迹数据，"
          f"其中丢弃了 {invalid_timestamps.value} 条无效时间戳数据，"
          f"丢弃了 {empty_value.value} 条空值数据点,"
          f"丢弃了 {over_time_range.value} 条时间小于五分钟或者大于两小时或者跨天的轨迹数据点")

    return trajectories
