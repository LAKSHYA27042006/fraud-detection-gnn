import torch


def save_model(model, filename):

    torch.save(
        model.state_dict(),
        f"saved_models/{filename}"
    )

    print(f"{filename} saved successfully")