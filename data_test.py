from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.create_graph_data import create_graph_data

features, edges, labels = load_data()

x, edge_index, y = preprocess_data(
    features,
    edges,
    labels
)

data = create_graph_data(
    x,
    edge_index,
    y
)

print(data)

print("\nData Summary")

print("Number of nodes:", data.num_nodes)

print("Number of edges:", data.num_edges)

print("Number of features:", data.num_node_features)