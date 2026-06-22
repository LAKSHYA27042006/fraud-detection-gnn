from torch_geometric.data import Data

import torch


def create_graph_data(x, edge_index, y):

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    valid_nodes = (y != -1)

    valid_indices = valid_nodes.nonzero(
        as_tuple=True
    )[0]

    n = len(valid_indices)

    train_size = int(0.8 * n)

    train_idx = valid_indices[:train_size]

    test_idx = valid_indices[train_size:]

    data.train_mask = torch.zeros(
        len(y),
        dtype=torch.bool
    )

    data.test_mask = torch.zeros(
        len(y),
        dtype=torch.bool
    )

    data.train_mask[train_idx] = True

    data.test_mask[test_idx] = True

    return data