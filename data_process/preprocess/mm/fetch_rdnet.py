import time
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import os
from os.path import join
from leuvenmapmatching.map.inmem import InMemMap
import osmnx as ox
from typing import List


# fetch chengdu
def fetch_map(city: str, bounds: List[float], save_path: str):
    print(f"Fetching map for {city}...")  # 添加打印信息
    if os.path.exists(join(save_path, f"{city}.graphml")):
        print(f"{city}.graphml already exists.")
        return
    north, south, east, west = bounds[3], bounds[1], bounds[2], bounds[0]  # 最大/小纬度，最大/小经度
    # 设置 Overpass API 端点
    ox.settings.overpass_endpoint = "https://overpass.openstreetmap.org/api/interpreter"

    # 创建 bbox 元组
    bbox = (north,south,east,west)

    if None in bbox:
        raise ValueError("Bounding box contains None values.")
    # 使用 bbox 作为单一参数传递给 graph_from_bbox
    g = ox.graph_from_bbox(north, south, east, west, network_type='drive')  # 获取城市道路网络

    # 保存图数据
    ox.save_graphml(g, join(save_path, f"{city}.graphml"))
    print(f"Saved {city} road network to {join(save_path, f'{city}.graphml')}")


    # # 设置备用的 Overpass API 服务器
    # ox.settings.overpass_url = "https://overpass-api.openstreetmap.fr/api/interpreter"

    # 尝试从 OpenStreetMap 获取城市道路网络
    # for attempt in range(retries):
    #     try:
    #         g = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    #         ox.save_graphml(g, join(save_path, f"{city}.graphml"))
    #         return  # 成功后退出
    #     except Exception as e:
    #         print(f"Attempt {attempt + 1} failed: {e}")
    #         if attempt < retries - 1:
    #             print("Retrying...")
    #             time.sleep(5)  # 暂停5秒后重试
    #         else:
    #             print("Max retries reached. Exiting.")
    #             raise
            
            
# build map
def build_map(city: str, map_path: str):
    print("Starting to load the graphml file...")
    g = ox.load_graphml(join(map_path, f"{city}.graphml"))
    print("Graph loaded successfully!")

    print("Converting to GeoDataFrames...")
    nodes_p, edges_p = ox.graph_to_gdfs(g, nodes=True, edges=True)
    print("Conversion to GeoDataFrames completed!")

    print("Plotting edges...")
    edges_p.plot()
    plt.savefig(join(map_path, "map.pdf"))
    plt.clf()

    print("Creating InMemMap...")
    map_con = InMemMap(name=f"map_{city}", use_latlon=True, use_rtree=False, index_edges=True, dir=map_path)
    print("InMemMap created successfully!")
 
    # construct road network
    nid_to_cmpct = dict()
    cmpct_to_nid = []
    # 将图中节点的原始id映射为紧凑的id，比如原始id为101，984，20，653；映射后为0,1,2,3
    for node_id, row in nodes_p.iterrows():
        if node_id not in nid_to_cmpct:
            nid_to_cmpct[node_id] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id)
        cid = nid_to_cmpct[node_id]
        map_con.add_node(cid, (row['y'], row['x'])) # 将映射后的id和经纬度信息添加到地图对象中
    # 将图中边的起终节点id映射为紧凑的id，比如原始id为(101, 984), (984, 101), (984, 20), (20, 984)，映射后为(0, 1), (1, 0), (1, 2), (2, 1)
    for node_id_1, node_id_2, _ in g.edges:
        if node_id_1 not in nid_to_cmpct:
            nid_to_cmpct[node_id_1] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id_1)
        if node_id_2 not in nid_to_cmpct:
            nid_to_cmpct[node_id_2] = len(cmpct_to_nid)
            cmpct_to_nid.append(node_id_2)
        cid1 = nid_to_cmpct[node_id_1]
        cid2 = nid_to_cmpct[node_id_2]
        map_con.add_edge(cid1, cid2)
        map_con.add_edge(cid2, cid1)
    map_con.dump() # 将地图对象保存到磁盘（map_path）
    return map_con


if __name__ == "__main__":
    city="chengdu"
    bounds = [104.0, 30.64, 104.15, 30.73]
    print("当前工作目录:", os.getcwd())
    save_path = "./sets_data/real/map"
    fetch_map(city, bounds, save_path)
    map_con = build_map(city, save_path)
