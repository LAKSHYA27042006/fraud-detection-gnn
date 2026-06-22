import networkx as nx
import pandas as pd


def detect_suspicious_networks(edges):

    G = nx.from_pandas_edgelist(

        edges,

        source="txId1",

        target="txId2"

    )

    components = list(

        nx.connected_components(G)

    )

    suspicious = []

    for idx, component in enumerate(components):

        size = len(component)

        if size >= 5:

            suspicious.append({

                "Network ID": idx + 1,

                "Transactions": size

            })

    suspicious_df = pd.DataFrame(

        suspicious

    )

    return suspicious_df