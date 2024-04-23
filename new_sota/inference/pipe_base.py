from transformers import pipeline
import torch


def get_pipe(model, user="Theoreticallyhugo"):
    """
    get pipe for huggingface model from huggingface repo
    args:
        model: model to load from huggingface repo
        user: whose model it is
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("loading pipeline")
    return pipeline("token-classification", model=f"{user}/{model}", device=device)
