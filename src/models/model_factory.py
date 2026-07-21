from transformers import DistilBertConfig, DistilBertModel
from src.utils.seed import set_seed


def create_model(hidden_size: int, seed: int = 42) -> DistilBertModel:
    """
    Create a custom DistilBERT model.

    Parameters
    ----------
    hidden_size : int
        Desired hidden dimension.
        Supported values should be divisible by 4.

    seed : int
        Random seed used for reproducible weight initialization.

    Returns
    -------
    DistilBertModel
        Newly initialized DistilBERT model.
    """

    if hidden_size <= 0:
        raise ValueError("hidden_size must be greater than zero.")

    if hidden_size % 4 != 0:
        raise ValueError(
            "hidden_size must be divisible by 4 because the model uses 4 attention heads."
        )

    set_seed(seed)

    config = DistilBertConfig(
        dim=hidden_size,
        hidden_dim=hidden_size * 4,
        n_heads=4,
        n_layers=6,
    )

    return DistilBertModel(config)
