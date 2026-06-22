import pandas as pd

from src.data_loader import load_data

from src.preprocessing import preprocess_data

from src.create_graph_data import create_graph_data

from src.train import train_model

from src.train_gat import train_gat

from src.evaluate import evaluate_model


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

print("Training GraphSAGE")

graphsage = train_model(data)

graphsage_metrics = evaluate_model(
    graphsage,
    data
)

print("\nTraining GAT")

gat = train_gat(data)

gat_metrics = evaluate_model(
    gat,
    data
)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "GraphSAGE": [

        graphsage_metrics["accuracy"],

        graphsage_metrics["precision"],

        graphsage_metrics["recall"],

        graphsage_metrics["f1_score"]

    ],

    "GAT": [

        gat_metrics["accuracy"],

        gat_metrics["precision"],

        gat_metrics["recall"],

        gat_metrics["f1_score"]

    ]
})

print(comparison)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)