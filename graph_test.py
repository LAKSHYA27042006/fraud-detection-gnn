from src.data_loader import load_data
from src.graph_builder import build_graph


features, edges, labels = load_data()

G = build_graph(edges)

print("Number of nodes :", G.number_of_nodes())

print("Number of edges :", G.number_of_edges())