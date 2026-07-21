import pytest

from src.models.model_factory import create_model


@pytest.mark.parametrize("hidden_size", [16, 32, 64, 128, 256])
def test_model_dimensions(hidden_size):
    model = create_model(hidden_size)

    assert model.config.dim == hidden_size
    assert model.config.hidden_dim == hidden_size * 4
    assert model.config.n_layers == 6
    assert model.config.n_heads == 4


def test_invalid_hidden_size():
    with pytest.raises(ValueError):
        create_model(30)
