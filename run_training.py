from src.data_loader import load_data

from src.preprocessing import preprocess_data

from src.create_graph_data import create_graph_data

from src.train import train_model

from src.evaluate import evaluate_model

from src.save_model import save_model

save_model(
    model,
    "graphsage_model.pth"
)
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

model = train_model(data)

evaluate_model(
    model,
    data
)