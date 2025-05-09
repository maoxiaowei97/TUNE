import matplotlib.pyplot as plt
import torch
plt.switch_backend("agg")
from data_process.preprocess.mm.refine_gps import get_trajectories
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
import multiprocessing
from multiprocessing import Manager, Pool, cpu_count
from tqdm import tqdm
import os
from os.path import join
import pickle
import h5py
import numpy as np
import networkx as nx
import pandas as pd
import folium


# # 全局共享统计变量管理
# def init_stats():
#     return {
#         'total_trajectories': 0,  # 总共的轨迹数据数量
#         'discarded_trajectories': 0,  # 丢弃的轨迹数量
#         'failed_matching_trajectories': 0,  # 匹配失败的轨迹数量
#         'short_trajectories': 0,  # 丢弃路径过短的轨迹数量
#     }

# 对给定的单条轨迹进行地图匹配
def map_single(trajectory, map_con, stats):
    stats['total_trajectories'] += len(trajectory)  # 统计每次传入的轨迹数量

    path = [[each[1], each[2]] for each in trajectory]

    matcher = DistanceMatcher(map_con,
                              # max_dist=400,
                              max_dist = 600,
                              # max_dist_init=400,
                              max_dist_init = 600,
                              min_prob_norm=0.2,
                              obs_noise=100,
                              obs_noise_ne=100,
                              dist_noise=100,
                              # max_lattice_width=20,
                              max_lattice_width=30,
                              non_emitting_states=False)

    # 匹配的状态序列，匹配的路径总长度
    states, match_length = matcher.match(path)

    # 匹配失败的情况
    if len(states) < len(path):
        stats['failed_matching_trajectories'] += len(trajectory)  # 路径匹配失败的丢弃
        stats['discarded_trajectories'] += len(trajectory)  # 丢弃轨迹
        return None

    # shrink
    states_shrinked = [states[0]]  # 取匹配状态列表 states 的第一个状态作为起始点
    link_points = [states[0][0], states[0][1]]  # link_points 依次保存路段连接点
    states_to_point = [[trajectory[0]]]  # states_to_point 保存每一个路段对应的轨迹点（可能不止一个）

    for i in range(1, len(states)):
        if states[i - 1][0] != states[i][0] or states[i - 1][1] != states[i][1]:
            assert states[i - 1][1] == states[i][0]  # 前一个状态的终止点和当前状态的起始点相同
            link_points.append(states[i][1])
            states_shrinked.append(states[i])
            states_to_point.append([trajectory[i]])  # 新的子列表，保存当前点
        else:  # 如果前一个状态和当前状态的起始点和终止点相同
            states_to_point[-1].append(trajectory[i])  # 将当前点添加到前一个路段的轨迹点中

    # 从状态序列中移除循环路径，生成非循环状态序列
    states_non_loop = []
    node_states = [a for a, b in states_shrinked] + [states_shrinked[-1][1]]  # 保存所有的节点状态
    show_pos = dict()  # 记录每个节点在 states_non_loop 中的位置
    for a in node_states:
        if a not in show_pos:
            show_pos[a] = len(states_non_loop)
            states_non_loop.append(a)
        else:
            for k in range(len(states_non_loop) - 1, show_pos[a], -1):
                last = states_non_loop.pop()
                show_pos.pop(last)

    # 路径过短，丢弃
    if len(states_non_loop) < 5:
        stats['short_trajectories'] += len(trajectory)  # 路径过短的丢弃
        stats['discarded_trajectories'] += len(trajectory)  # 丢弃路径
        return None

    return (link_points, states_shrinked, states_to_point, states_non_loop)


# 对给定的一批轨迹进行地图匹配
def map_batch(pid, trajectories, city, map_path, stats):
    map_con = InMemMap.from_pickle(join(map_path, f"map_{city}.pkl"))
    trajectories_mapped = []

    for i in tqdm(range(len(trajectories)), ncols=80, position=pid):
        states_to_point_idx_states = map_single(trajectories[i], map_con, stats)
        if states_to_point_idx_states:
            trajectories_mapped.append(states_to_point_idx_states)

    return trajectories_mapped


# 对给定的所有轨迹数据进行地图匹配
def mapmatching(date, city, raw_traj_path, map_path):
    trajectories = get_trajectories(date, raw_traj_path)
    print("已经获得了处理后的轨迹数据")
    trajectories_mapped = []

    # 如果没有轨迹数据（可能因为全部数据都被丢弃），跳过当前文件
    if not trajectories:
        print(f"警告：日期 {date} 的轨迹数据为空，跳过此文件(processed_{date}.csv)")
        return []

    # 使用 Manager 来管理共享变量
    with multiprocessing.Manager() as manager:
        stats = manager.dict(
            {'total_trajectories': 0,  # 总共的轨迹数据数量
             'discarded_trajectories': 0,  # 丢弃的轨迹数量
             'failed_matching_trajectories': 0,  # 匹配失败的轨迹数量
             'short_trajectories': 0})  # 丢弃路径过短的轨迹数量

        n_process = min(int(cpu_count()) + 1, 20)
        trajectories_mapped_batch_mid = []

        with multiprocessing.Pool(processes=n_process) as pool:
            err = lambda err: print(err)
            batch_size = (len(trajectories) + n_process - 1) // n_process

            for i in range(0, len(trajectories), batch_size):
                pid = i // batch_size
                trajectory_mapped_batch = pool.apply_async(map_batch, (
                    pid, trajectories[i:i + batch_size], city, map_path, stats))
                trajectories_mapped_batch_mid.append(trajectory_mapped_batch)

            for each in trajectories_mapped_batch_mid:
                trajectories_mapped.extend(each.get())

        # 打印统计结果
        print(f"总共处理了 {stats['total_trajectories']} 条数据，其中丢弃了 {stats['discarded_trajectories']} 条数据，"
              f"路径匹配失败丢弃了 {stats['failed_matching_trajectories']} 条数据，"
              f"路径过短丢弃了 {stats['short_trajectories']} 条数据")

    return trajectories_mapped


# 获取已经匹配好的路径
def get_matched_path(date, city, traj_path, map_path, raw_path):
    target_path = join(traj_path, f"traj_mapped_{city}_{date}.pkl")
    if os.path.exists(target_path):
        print("loading...")
        return pickle.load(open(target_path, "rb"))
    trajectories_mapped = mapmatching(date, city, raw_path, map_path)
    print("writing...")
    pickle.dump(trajectories_mapped, open(target_path, "wb"))
    print("write complete!")
    return trajectories_mapped



def process_gps_and_graph(city, map_path, data_path, raw_path, traj_path):
    name = city
    map_con = InMemMap.from_pickle(join(map_path, f"map_{city}.pkl")) # 加载地图对象
    # calculate G and A
    target_g_path = join(data_path, f"{name}_G.pkl") # 保存地图对象
    G = nx.Graph() # 创建一个空的无向图
    node_attrs = [(cid, {"lat": lat, "lng": lng}) for cid, (lat, lng) in map_con.all_nodes()]
    G.add_nodes_from(node_attrs)
    G.add_edges_from([(a, b) for a, _, b, _ in map_con.all_edges()])
    # 创建A作为图G的邻接矩阵
    n = G.number_of_nodes()
    A = torch.zeros([n, n], dtype=torch.float64)
    for a, b in G.edges:
        A[a, b] = 1.
        A[b, a] = 1.
    pickle.dump(G, open(target_g_path, "wb"))
    torch.save(A, join(data_path, f"{name}_A.ts"))
    # 从raw_path中读取gps文件,提取出含有gps的文件名放入gps_file_list
    gps_file_list = list(os.listdir(raw_path))
    gps_file_list.sort()
    gps_file_list = [each for each in gps_file_list if "gps" in each]
    h5_file = join(data_path, f"{name}_h5_paths.h5")
    with h5py.File(h5_file, "w") as f:
        for gps_file in gps_file_list[:1]:
            date = gps_file[4:] # 从文件名的第4个字符开始截取日期，如gps_20161101.txt截取出20190701
            print("#####", date)
            f.create_group(date)
            trajectories_mapped = get_matched_path(date, city, traj_path, map_path, raw_path)
            # shrink 
            state_lengths, states = [], []

            for link_points, states_shrinked, states_to_point, states_non_loop in trajectories_mapped:
                state_lengths.append(len(states_non_loop))
                states.extend(states_non_loop)

            # calcluate prefix sum
            state_prefix = np.zeros(shape=len(state_lengths) + 1, dtype=np.int64)
            for k, L in enumerate(state_lengths):
                state_prefix[k + 1] = state_prefix[k] + L
            
            # length_info
            # pad all in one
            f[date].create_dataset("state_prefix", data=np.array(state_prefix))
            f[date].create_dataset("states", data=np.array(states))
    
    # calculate V
    target_v_path = join(data_path, f"{name}_v_paths.csv")
    vs = []
    for gps_file in gps_file_list[:1]:
        date = gps_file[4:]
        trajectories_mapped = get_matched_path(date, city, traj_path, map_path, raw_path)
        # (link_points, states_shrinked, states_to_point, states_non_loop)
        non_loops  = [each[-1] for each in trajectories_mapped]
        n_samples = len(non_loops)
        v_np = np.zeros([n_samples, n])
        for k, non_loop in enumerate(non_loops):
            v_np[k, non_loop] = 1.
            v_np[k, non_loop[0]] = 2.
        vs.append(v_np)
    # generate V
    v_data = np.concatenate(vs, axis=0)
    np.savetxt(target_v_path, v_data, delimiter=',', fmt='%d')


def draw_trajectories(trajectories_mapped, G, html_path):
    # 检查目标目录是否存在，不存在则创建
    directory = os.path.dirname(html_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 从匹配好的路径中提取经纬度
    multiple_locs = []

    # 遍历所有的轨迹
    for link_points, states_shrinked, states_to_point, states_non_loop in trajectories_mapped:
        locs = []

        # 遍历轨迹的每个路段
        for segment in states_to_point:
            for point in segment:  # 遍历路段中的每个点
                lat, lng = point[1], point[2]  # 提取纬度和经度
                locs.append([lat, lng])  # 将经纬度加入到轨迹的列表中
        multiple_locs.append(locs)  # 将一条轨迹添加到多个轨迹的列表中

    # 计算地图中心
    cen_lng, cen_lat, cnt = 0, 0, 0
    for series in multiple_locs:
        cnt += len(series)
        for y, x in series:
            cen_lat += y
            cen_lng += x

    if cnt == 0:
        print(f"警告：当前处理文件的轨迹数据为空，跳过绘制操作")
        return

    # 创建地图
    m = folium.Map([cen_lat / cnt, cen_lng / cnt], zoom_start=13, attr='default',
                   tiles='https://tile.openstreetmap.org/{z}/{x}/{y}.png')

    # 绘制每条轨迹
    for k, locations in enumerate(multiple_locs):
        color = "red"  # 默认为红色，如果需要的话可以调整为不同的颜色
        folium.PolyLine(locations, weight=5, color=color, opacity=0.7).add_to(m)

        # 为每条轨迹添加起点和终点标记
        if locations:
            folium.CircleMarker(locations[0], radius=5, fill=True, opacity=1., color="blue", fill_color="blue",
                                fill_opacity=1., popup='<b>Starting Point</b>').add_to(m)
            folium.CircleMarker(locations[-1], radius=5, fill=True, opacity=1., color="green", fill_color="green",
                                fill_opacity=1., popup='<b>End Point</b>').add_to(m)

    # 保存地图为HTML文件
    m.save(html_path)
    print(f"地图已保存为 {html_path}")


def process_gps_and_graph_WTR(city, map_path, data_path, raw_path, traj_path):
    name = city
    # 加载地图对象
    map_con = InMemMap.from_pickle(join(map_path, f"map_{city}.pkl"))

    # 计算图G和邻接矩阵A
    target_g_path = join(data_path, f"{name}_G.pkl")  # 保存图对象
    G = nx.Graph()  # 创建一个空的无向图
    node_attrs = [(cid, {"lat": lat, "lng": lng}) for cid, (lat, lng) in map_con.all_nodes()]
    G.add_nodes_from(node_attrs)
    G.add_edges_from([(a, b) for a, _, b, _ in map_con.all_edges()])

    # 创建邻接矩阵A
    n = G.number_of_nodes()
    A = torch.zeros([n, n], dtype=torch.float64)
    for a, b in G.edges:
        A[a, b] = 1.
        A[b, a] = 1.
    pickle.dump(G, open(target_g_path, "wb"))
    torch.save(A, join(data_path, f"{name}_A.ts"))

    # 读取处理后的文件列表
    traj_file_list = list(os.listdir(raw_path))
    traj_file_list.sort()

    # 打印文件列表
    print("文件列表：", traj_file_list)

    # 只选择以 "xian" 开头的文件
    traj_file_list = [each for each in traj_file_list if each.startswith("xian") and each.endswith(".csv")]

    print("选择的文件列表：", traj_file_list)


    for traj_file in traj_file_list:
        date = traj_file.split("_")[3]  # 提取日期部分
        date = date.split(".")[0]  # 去掉可能的扩展名（通常这里没有扩展名）
        print(f"#####################{date}#####################")

        # 获取匹配的路径
        trajectories_mapped = get_matched_path(date, city, traj_path, map_path, raw_path)

        # 为每个日期创建一个DataFrame并保存
        data_list = []
        for link_points, states_shrinked, states_to_point, states_non_loop in trajectories_mapped:
            for i, state in enumerate(states_shrinked):
                # 在此构建数据
                for point in states_to_point[i]:
                    # 提取所需的列
                    timestamp = point[0]  # 时间戳
                    lat = point[1]  # 纬度
                    lon = point[2]  # 经度
                    traj_id = point[3]  # 轨迹编号
                    start_segment = states_shrinked[i][0]  # 起点路段编号
                    end_segment = states_shrinked[i][1]  # 终点路段编号

                    # 将数据添加到列表中
                    data_list.append([traj_id, timestamp, lon, lat, start_segment, end_segment])

        # HDF5文件路径
        h5_file = f"./mapped_traj/xian/mapped_traj_{name}_{date}.h5"
        # CSV文件路径
        csv_file = f"./mapped_traj/xian/mapped_traj_{name}_{date}.csv"

        # 创建 DataFrame
        df = pd.DataFrame(data_list,
                          columns=['轨迹编号', '时间戳', '轨迹点经度', '轨迹点纬度', '起点路段编号', '终点路段编号'])
        df.to_hdf(h5_file, key='df')
        # # 保存数据为 CSV 文件
        # df.to_csv(csv_file, index=False, mode='w', header=True)

        # print(f"data saved to {h5_file} and {csv_file}")
        print(f"data saved to {h5_file}")

        # 绘制处理后的轨迹
        html_path = f"./figs/processed_trajectories_{name}_{date}.html"
        draw_trajectories(trajectories_mapped, G, html_path)
