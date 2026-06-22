import pandas as pd


def load_data():

    features = pd.read_csv(
        "dataset/raw/elliptic_txs_features.csv",
        header=None
    )

    edges = pd.read_csv(
        "dataset/raw/elliptic_txs_edgelist.csv"
    )

    labels = pd.read_csv(
        "dataset/raw/elliptic_txs_classes.csv"
    )

    return features, edges, labels