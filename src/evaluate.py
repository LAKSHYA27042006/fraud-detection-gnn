import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
        model,
        data
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data = data.to(device)

    model.eval()

    with torch.no_grad():

        out = model(
            data.x,
            data.edge_index
        )

        pred = out.argmax(
            dim=1
        )

    y_true = data.y[
        data.test_mask
    ].cpu()

    y_pred = pred[
        data.test_mask
    ].cpu()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

    print("\nResults")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")