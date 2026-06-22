from src.data_loader import load_data

features, edges, labels = load_data()

print("Features Shape :", features.shape)

print("Edges Shape :", edges.shape)

print("Labels Shape :", labels.shape)

print("\nFeatures")

print(features.head())

print("\nEdges")

print(edges.head())

print("\nLabels")

print(labels.head())