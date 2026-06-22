import torch

import torch.nn.functional as F

from models.graphsage_model import GraphSAGE


def train_model(data):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data = data.to(device)

    model = GraphSAGE(
        data.num_node_features
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train_mask = data.train_mask

    for epoch in range(20):

        model.train()

        optimizer.zero_grad()

        out = model(
            data.x,
            data.edge_index
        )

        loss = F.cross_entropy(
    out[train_mask],
    data.y[train_mask]
)

        loss.backward()

        optimizer.step()

        print(
            f"Epoch {epoch+1} | Loss : {loss.item():.4f}"
        )

    return model