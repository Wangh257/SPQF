from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def _activation(name: str) -> Callable[[], nn.Module]:
    choices: dict[str, Callable[[], nn.Module]] = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"不支持的激活函数 {name!r}，可选 {sorted(choices)}") from exc


class PhaseToDepthMLP(nn.Module):
    """Direct MLP: ``(u_n,v_n,Phi_n[,sinPhi,cosPhi,A]) -> zn``。"""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        hidden_layers: int = 5,
        activation: str = "silu",
        skip_layer: int | None = 3,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or hidden_layers <= 0:
            raise ValueError("网络维度和层数必须大于 0")
        if skip_layer is not None and not 0 < skip_layer < hidden_layers:
            raise ValueError("skip_layer 必须位于隐藏层之间，或设为 null")
        make_activation = _activation(activation)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.skip_layer = skip_layer
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer_index in range(hidden_layers):
            layer_input = input_dim if layer_index == 0 else hidden_dim
            if skip_layer is not None and layer_index == skip_layer:
                layer_input += input_dim
            self.layers.append(nn.Linear(layer_input, hidden_dim))
            self.activations.append(make_activation())
        self.output = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in list(self.layers) + [self.output]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        # 让初始深度输出接近归一化范围中心，但不使用 sigmoid 限制输出。
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.output.bias, 0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"网络期望最后一维为 {self.input_dim}，实际为 {features.shape[-1]}")
        original = features
        hidden = features
        for layer_index, (layer, activation) in enumerate(
            zip(self.layers, self.activations)
        ):
            if self.skip_layer is not None and layer_index == self.skip_layer:
                hidden = torch.cat((hidden, original), dim=-1)
            hidden = activation(layer(hidden))
        return self.output(hidden).squeeze(-1)

    def config_dict(self) -> dict[str, int | str | None]:
        activation = self.activations[0].__class__.__name__.lower()
        if activation == "silu":
            activation_name = "silu"
        else:
            activation_name = activation
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "activation": activation_name,
            "skip_layer": self.skip_layer,
        }
