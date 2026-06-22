import torch

import torch.nn.functional as F

from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.gat1 = GATConv(
            input_dim,
            128,
            heads=4
        )

        self.gat2 = GATConv(
            512,
            64,
            heads=1
        )

        self.fc = torch.nn.Linear(
            64,
            2
        )

    def forward(
            self,
            x,
            edge_index
    ):

        x = self.gat1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.gat2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.fc(x)

        return x