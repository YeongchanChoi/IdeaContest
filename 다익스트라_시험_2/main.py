from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from math import radians, cos, sin, asin, sqrt
from fastapi.responses import FileResponse

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# Pydantic 모델 정의
class RouteRequest(BaseModel):
    startNode: int
    desNode: int
    timeOfDay: str

# 시간대별 밀집도 데이터 로드
file_path = 'time_value.csv'
time_value_df = pd.read_csv(file_path)
time_value_dict = time_value_df.set_index('time').to_dict()['value']

# Haversine 공식을 사용하여 두 지점 간의 거리 계산
def haversine(lon1, lat1, lon2, lat2):
    # 지구 반경 (km)
    R = 6371.0
    # 라디안으로 변환
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # 경도와 위도 차이
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine 공식
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    distance = R * c
    return distance

# 그래프 생성 및 초기화
G = nx.Graph()
nodes = [
   (30, (37.447720, 126.659337)),
(19, (37.451384, 126.650712)),
(2, (37.450559, 126.653770)),
(4, (37.451125, 126.654314)),
(38, (37.450831, 126.654524)),
(37, (37.449158, 126.655874)),
(35, (37.449663, 126.657140)),
(34, (37.450641, 126.657802)),
(33, (37.451391, 126.654898)),
(32, (37.449339, 126.655378)),
(31, (37.450036, 126.653568)),
(29, (37.450624, 126.656893)),
(5, (37.450309, 126.654807)),
(6, (37.450769, 126.655774)),
(7, (37.450837, 126.655059)),
(8, (37.450895, 126.656019)),
(10, (37.449839, 126.655449)),
(11, (37.449940, 126.654944)),
(12, (37.448956, 126.654645)),
(13, (37.449245, 126.653978)),
(14, (37.449546, 126.652967)),
(15, (37.450477, 126.652943)),
(17, (37.451031, 126.651936)),
(20, (37.448254, 126.652253)),
(21, (37.447377, 126.654205)),
(22, (37.447320, 126.655358)),
(23, (37.448153, 126.655990)),
(24, (37.448612, 126.655868)),
(25, (37.449955, 126.656393)),
(26, (37.449053, 126.656355)),
(28, (37.450236, 126.656968))
]

edges = [
  (2, 4), (4, 33), (33, 38), (38, 7), (38, 6), (38, 5), (33, 8), (8, 10), (8, 29), (8, 25), (25, 28), (28, 29), (25, 29),
(8, 34), (34, 35), (25, 26), (26, 35), (26, 37), (37, 24), (23, 24), (30, 35), (24, 22), (24, 21), (37, 32), (37, 25),
(10, 32), (32, 21), (21, 20), (20, 14), (14, 13), (13, 12), (12, 24), (20, 15), (14, 31), (31, 2), (31, 15), (31, 32),
(15, 17), (17, 19), (17, 4), (31, 38), (31, 11), (11, 32)
]

# 노드 추가
for node, (lat, lon) in nodes:
    G.add_node(node, pos=(lon, lat))

# 간선 추가 및 거리 계산
for u, v in edges:
    lat1, lon1 = G.nodes[u]['pos']
    lat2, lon2 = G.nodes[v]['pos']
    distance = haversine(lon1, lat1, lon2, lat2)
    G.add_edge(u, v, distance=distance)

def update_weights_for_time(graph, time):
    # 시간에 따른 밀집도 값 가져오기
    density_value = time_value_dict.get(time, 0)

    # 밀집도에 따른 가중치 설정
    if density_value < 10:
        density_weight = 1
    elif density_value < 30:
        density_weight = 2
    else:
        density_weight = 3

    # 각 간선에 대해 거리와 밀집도 가중치 합산
    for u, v, data in graph.edges(data=True):
        data['weight'] = data['distance'] + density_weight

def dijkstra(graph, start, end):
    if start not in graph or end not in graph:
        raise ValueError("Start or end node not in graph.")
    update_weights_for_time(graph, '08:00')  # 예시로 '08:00'을 사용
    path = nx.shortest_path(graph, source=start, target=end, weight='weight')
    return path

def plot_path(graph, path):
    pos = nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700, edge_color='gray', edge_cmap=plt.cm.Blues)
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color='red')
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.title("Shortest Path")
    plt.savefig("shortest_path.png")
    plt.close()

@app.post("/find-route/")
def find_route(request: RouteRequest):
    start_node = request.startNode
    des_node = request.desNode
    time_of_day = request.timeOfDay

    # 시작 노드와 도착 노드가 30 이상인지 검사
    if start_node >= 30 or des_node >= 30:
        raise HTTPException(status_code=400, detail="Start and destination nodes must be less than 30.")

    try:
        update_weights_for_time(G, time_of_day)
        path = dijkstra(G, start_node, des_node)
        plot_path(G, path)
        return {"message": "Success", "path": path, "image": "shortest_path.png"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
