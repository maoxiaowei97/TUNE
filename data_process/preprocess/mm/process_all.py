import sys
import os

# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前脚本所在的目录
current_dir = os.path.dirname(current_file_path)

# 设置工作目录为项目根目录
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# 切换到项目根目录
os.chdir(project_root_dir)

# 打印当前工作目录，确认是否更改成功
print("当前工作目录:", os.getcwd())
def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()
sys.path.append(ws)

from loader.preprocess.mm.fetch_rdnet import fetch_map, build_map
from loader.preprocess.mm.mapmatching import process_gps_and_graph, process_gps_and_graph_WTR



if __name__ == "__main__":
    
    data_path = "./data/"
    city = "xian"
    bounds = [108.9, 34.20, 109.01, 34.28]
    map_path = "./data/real2/map"
    # 获取西安的地图数据
    fetch_map(city, bounds, map_path)
    # 构建地图对象
    map_con = build_map(city, map_path)

    raw_path = "./data/real2/raw"
    traj_path = "./data/real2/trajectories"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    process_gps_and_graph_WTR(city, map_path, data_path, raw_path, traj_path)


    # process real 成都
    city = "chengdu"
    bounds = [104.0, 30.64, 104.15, 30.73]
    map_path = "./sets_data/real/map"
    fetch_map(city, bounds, map_path)
    print("fetch done!")

    map_con = build_map(city, map_path)
    print("map done!")

    raw_path = "./sets_data/real/raw"
    traj_path = "./sets_data/real/trajectories"
    # process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    process_gps_and_graph_WTR(city, map_path, data_path, raw_path, traj_path)
    print("chengdu done!")



# ======================== 输入数据和输出数据格式 ============================
# 输入数据：processed_{date}.csv 文件
# 五列：订单号、司机号、时间戳、经度、纬度(订单号即为轨迹编号)
# OrderId,CarId,Timestamp,Lng,Lat
# f80ce851e2f810a0a5b90672f5348ad2,6d80fea7c664ea5ced91465c5ad62622,1540828237,108.96372,34.27002

# 输出数据：cd_mapped_traj_details_xian_{date}.h5 文件和cd_mapped_traj_details_{date}.csv 文件
# 六列
# 轨迹编号,时间戳,轨迹点经度,轨迹点纬度,起点路段编号,终点路段编号
# 00276334c7e4030c3ae6f2fa8eaf6b85,1539665169,108.94513,34.2358,441,3528

