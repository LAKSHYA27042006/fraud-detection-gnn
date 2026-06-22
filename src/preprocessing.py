import pandas as pd
import torch


def preprocess_data(features, edges, labels):

    # ----------------------
    # Create node mapping
    # ----------------------

    tx_ids = features.iloc[:, 0].values

    node_mapping = {
        tx_id: idx
        for idx, tx_id in enumerate(tx_ids)
    }

    # ----------------------
    # Node features (x)
    # ----------------------

    x = torch.tensor(
        features.iloc[:, 1:].values,
        dtype=torch.float
    )

    # ----------------------
    # Edge index
    # ----------------------

    edges["source"] = edges["txId1"].map(node_mapping)

    edges["target"] = edges["txId2"].map(node_mapping)

    edges = edges.dropna()

    edge_index = torch.tensor(
        edges[["source", "target"]].values.T,
        dtype=torch.long
    )

    # ----------------------
    # Labels (y)
    # ----------------------

    labels = labels.set_index("txId")

    y = []

    for tx_id in tx_ids:

        label = labels.loc[tx_id, "class"]

        if label == "1":

            y.append(1)

        elif label == "2":

            y.append(0)

        else:

            y.append(-1)

    y = torch.tensor(y)

    return x, edge_index, y