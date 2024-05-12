import matplotlib.pyplot as plt
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
from haversine import haversine, nodes, edges

app = FastAPI()
class RouteRequest(BaseModel):
    start_node: int
    end_node: int
    day_of_week: str
    time_of_day: str
    @validator('day_of_week')
    def convert_day_of_week(cls, v):
        days_mapping = {
            "월요일": "Monday",
            "화요일": "Tuesday",
            "수요일": "Wednesday",
            "목요일": "Thursday",
            "금요일": "Friday"
        }
        if v not in days_mapping:
            raise ValueError("Invalid day of week")
        return days_mapping[v]
    @validator('start_node', 'end_node')
    def check_node_range(cls, value):
        if value >= 30:
            raise ValueError('노드 번호는 30 미만이어야 합니다.')
        return value

G = nx.Graph()
congested_edges = [(2, 4), (31, 2), (31, 32), (25, 37), (33, 8)]
for node1, node2 in edges:
    dist = haversine(*nodes[node1], *nodes[node2])
    G.add_edge(node1, node2, weight=dist)
def get_congestion_multiplier(value, day):
    """ 요일별 혼잡도 가중치 계산 """
    thresholds = {
        '월': (0.424383, 0.140647),
        '화': (0.328176, 0.183959),
        '수': (0.347222, 0.162602),
        '목': (0.3884494, 0.1626016),  
        '금': (0.243904, 0.1103753)
    }
    high, medium = thresholds[day]
    if value >= high:
        return 5
    elif value >= medium:
        return 3
    else:
        return 0
def update_weights(G, day_of_week, time_of_day):
    """ 그래프 간선 가중치 업데이트 """
    day_of_week_kr = {'Monday': '월요일', 'Tuesday': '화요일', 'Wednesday': '수요일', 'Thursday': '목요일', 'Friday': '금요일'}
    for edge in G.edges():
        distance_weight = G[edge[0]][edge[1]]['weight']
        congestion_multiplier = 0
        if edge in congested_edges:
            if edge in (33, 8):
                building = "2호관"
            elif edge in [(31, 2), (2,4)]:
                building = "6주"
            elif edge == (25, 37):
                building = "비플"
            elif edge == (31, 32):
                building = "본관"
            else:  # 이 외의 경우, 해당 간선은 존재하지 않음
                continue
            filename = f"{building}_{day_of_week_kr[day_of_week]}.csv"
            try:
                congestion_data = pd.read_csv(filename).set_index('time')
                try:
                    congestion_value = congestion_data.loc[time_of_day, 'value']
                    congestion_value = float(congestion_value)  # 여기에 타입 변환 추가
                except KeyError:
                    congestion_value = 0.0
                congestion_multiplier = get_congestion_multiplier(congestion_value, day_of_week_kr[day_of_week][0])
            except FileNotFoundError:
                pass  # 파일이 없으면 혼잡도 가중치를 0으로 유지
        # 비율에 맞춰 가중치 재계산
        adjusted_distance_weight = distance_weight * 0.6
        adjusted_congestion_weight = congestion_multiplier * 0.4
        G[edge[0]][edge[1]]['weight'] = adjusted_distance_weight + adjusted_congestion_weight
def find_shortest_path(G, start_node, end_node, day_of_week, time_of_day):
    """ 최단 경로 계산 """
    update_weights(G, day_of_week, time_of_day)
    return nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
@app.post("/route/")
async def get_route(request: RouteRequest):
    path = find_shortest_path(G, request.start_node, request.end_node, request.day_of_week, request.time_of_day)
    draw_graph(G, path=path)
    return {"route": path}
def draw_graph(G, path=None):
    pos = {node: (coord[1], coord[0]) for node, coord in nodes.items()}
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color='gray')
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.show()
