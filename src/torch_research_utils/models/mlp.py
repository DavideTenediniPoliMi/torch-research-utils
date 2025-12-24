from __future__ import annotations

from typing import Literal

import torch
from torch import nn

LayerConfig = dict[
    str,
    int | str | bool | float | None,
]


class MLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) builder.

    This module builds a network from a list of layer configurations.
    Global arguments (activation, norm, etc.) act as DEFAULTS
    for all layers, but can be overridden by any layer's specific
    config.

    ---
    Mode 1: Simple (using `units_list`)
    Provide a list of layer sizes. Global defaults will be applied. There is
    a QoL flag `activate_last` to disable the last layer's activation.
    >>> mlp = MLP(
    ...     input_dim=784,
    ...     units_list=[128, 64, 10],
    ...     activation="relu",
    ...     norm="layer",
    ...     activate_last=False,
    ... )

    ---
    Mode 2: Advanced (using `layer_configs`)
    Provide a list of config dicts for each layer. Any
    unspecified keys will be filled by the global defaults.
    >>> mlp = MLP(
    ...     input_dim=784,
    ...     activation="relu",  # Default activation
    ...     norm="layer",  # Default norm
    ...     layer_configs=[
    ...         {"units": 128},  # Gets relu, layer
    ...         {"units": 64, "activation": "silu"},  # Overrides act
    ...         {"units": 10, "norm": None, "activation": "softmax"},
    ...     ],
    ... )

    Args:
        input_dim (int | None): Dimensionality of the input features.
            If None, `nn.LazyLinear` is used for the first layer,
            and the input shape will be inferred.

        -- Config (Must provide one) --
        units_list (list[int] | None, optional): A simple list of output
            dimensions for each layer. Mutually exclusive with `layer_configs`.
        layer_configs (list[LayerConfig] | None, optional): A list of
            dictionaries for fine-grained, per-layer control. Mutually
            exclusive with `units_list`.

        -- Global Defaults (for all layers) --
        activation (str | nn.Module | None, optional): The default activation
            function for all layers. Defaults to "relu".
        norm (Literal["batch", "layer"] | None, optional): The default
            normalization for all layers. Defaults to None.
        dropout (float | None, optional): The default dropout probability.
            Defaults to None.
        use_bias (bool, optional): The default for whether Linear layers
            should have a bias. Defaults to True.
        norm_first (bool, optional): The default layer order.
            If True: (Linear -> Norm -> Act -> Dropout).
            Defaults to True.
        activate_last (bool, optional): QoL flag *only* for `units_list`
            mode. If False, the last layer's activation will be set to None,
            overriding the global default. Defaults to False.

        -- NEW: Input Processing --
        flatten_input (bool, optional): If True, any input tensor with
            `dim() > 2` will be flattened (from `start_dim=1`) before
            the first layer. Defaults to False.
    """

    def __init__(
        self,
        input_dim: int | None,
        # --- Config (Must provide one) ---
        layer_configs: list[LayerConfig] | None = None,
        units_list: list[int] | None = None,
        # --- Global Defaults (apply to all configs) ---
        activation: str | nn.Module | None = "relu",
        norm: Literal["batch", "layer"] | None = None,
        dropout: float | None = None,
        use_bias: bool = True,
        norm_first: bool = True,
        # --- QoL flag for simple mode only ---
        activate_last: bool = False,
    ) -> None:
        super().__init__()

        # --- 1. Store Input-Handling Fields ---
        self.input_dim = input_dim

        # --- 2. Validate and Establish Base Configs ---
        if (layer_configs is None) == (units_list is None):
            raise ValueError(
                "Must provide exactly one of 'layer_configs' or 'units_list'."
            )

        base_configs: list[LayerConfig]

        if units_list:
            # Simple mode: build a partial config list
            base_configs = [{"units": u} for u in units_list]

            # Apply 'activate_last' logic
            if not activate_last and base_configs:
                # By setting 'activation': None, we explicitly override
                # the global default for the last layer.
                base_configs[-1]["activation"] = None
        else:
            # Advanced mode: just copy the user's configs
            base_configs = [cfg.copy() for cfg in layer_configs]  # type: ignore

        # --- 3. Build the Network ---
        self.net = self._build_network(
            input_dim=self.input_dim,
            layer_configs=base_configs,
            default_activation=activation,
            default_norm=norm,
            default_dropout=dropout,
            default_use_bias=use_bias,
            default_norm_first=norm_first,
        )

    def _build_network(
        self,
        input_dim: int | None,
        layer_configs: list[LayerConfig],
        # Global defaults passed from __init__
        default_activation: str | nn.Module | None,
        default_norm: Literal["batch", "layer"] | None,
        default_dropout: float | None,
        default_use_bias: bool,
        default_norm_first: bool,
    ) -> nn.Sequential:
        """
        The main builder loop.

        Consumes the `layer_configs` list and fills in missing
        values with the provided global defaults.
        """
        layers: list[nn.Module] = []
        current_dim = input_dim

        for i, config in enumerate(layer_configs):
            # --- 1. Get Layer Configuration ---
            if "units" not in config:
                raise ValueError(f"Layer {i} config missing 'units' key.")

            units = int(config["units"])

            # --- Use per-layer value IF KEY EXISTS, else use global default ---
            use_bias = bool(config.get("use_bias", default_use_bias))
            activation_config = config.get("activation", default_activation)
            norm_type = config.get("norm", default_norm)
            dropout_p = float(config.get("dropout", default_dropout or 0.0))
            norm_first = bool(config.get("norm_first", default_norm_first))

            # --- 2. Build Layers in Order ---

            # --- NEW: Handle LazyLinear for the first layer ---
            linear_layer: nn.Linear | nn.LazyLinear
            if i == 0 and current_dim is None:
                # Use LazyLinear if input_dim is not specified
                linear_layer = nn.LazyLinear(units, bias=use_bias)
            else:
                # We must have a valid dimension
                if not isinstance(current_dim, int):
                    raise RuntimeError(
                        f"Layer {i} has no input dimension. "
                        "This should not happen if input_dim is not None."
                    )
                linear_layer = nn.Linear(current_dim, units, bias=use_bias)

            norm_layer = None
            if norm_type == "layer":
                norm_layer = nn.LayerNorm(units)
            elif norm_type == "batch":
                # BatchNorm1d expects [B, C] or [B, C, L]
                norm_layer = nn.BatchNorm1d(units)

            activation_layer = create_activation_fn(
                activation_config,
                require=False,  # type: ignore
            )

            dropout_layer = nn.Dropout(dropout_p) if dropout_p > 0 else None

            # --- 3. Add to Module List in specified order ---
            if norm_first:
                layers.append(linear_layer)
                if norm_layer:
                    layers.append(norm_layer)
                if activation_layer:
                    layers.append(activation_layer)
                if dropout_layer:
                    layers.append(dropout_layer)
            else:
                layers.append(linear_layer)
                if activation_layer:
                    layers.append(activation_layer)
                if norm_layer:
                    layers.append(norm_layer)
                if dropout_layer:
                    layers.append(dropout_layer)

            current_dim = units  # Output of this block is input to next

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass the input through the sequential network."""
        return self.net(x)

    def __repr__(self) -> str:
        """Provide a clean representation."""
        # The default nn.Sequential repr is already very good
        return f"MLP(\n  {self.net}\n)"
