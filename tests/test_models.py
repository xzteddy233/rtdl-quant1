import torch

from rtdl_quant.models import MLP, MLPConfig, ResNet, ResNetConfig
from rtdl_quant.models.wrappers import FTTransformerModel, ResNetModel


def test_mlp_output_shape() -> None:
    model = MLP(MLPConfig(d_in=5, hidden_dims=(8, 4), dropout=0.0))
    assert model(torch.randn(3, 5)).shape == (3,)


def test_resnet_output_shape_and_gradient() -> None:
    model = ResNet(
        ResNetConfig(
            d_in=5,
            depth=2,
            width=8,
            hidden_factor=2.0,
            dropout_first=0.0,
            dropout_second=0.0,
        )
    )
    output = model(torch.randn(4, 5))
    output.sum().backward()
    assert output.shape == (4,)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_official_resnet_wrapper_builds() -> None:
    model = ResNetModel(
        d_in=5,
        n_blocks=1,
        d_block=8,
        d_hidden_multiplier=2.0,
        epochs=1,
    )
    assert model.module(torch.randn(3, 5)).shape == (3, 1)


def test_ft_transformer_wrapper_builds() -> None:
    model = FTTransformerModel(
        n_num_features=5,
        n_blocks=1,
        d_block=32,
        epochs=1,
    )
    assert model.module(torch.randn(3, 5)).shape == (3, 1)
