from src.data_loader import load_data

from src.preprocessing import preprocess_data


features, edges, labels = load_data()

x, edge_index, y = preprocess_data(
    features,
    edges,
    labels
)

print("X Shape :", x.shape)

print("Edge Index Shape :", edge_index.shape)

print("Y Shape :", y.shape)

print("\nUnique Labels")

print(y.unique())