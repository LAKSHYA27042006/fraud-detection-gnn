import networkx as nx


def build_sample_graph(edges):

    sample = edges.head(300)

    G = nx.from_pandas_edgelist(
        sample,
        source="txId1",
        target="txId2"
    )

    return G