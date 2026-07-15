"""Optimizer construction for AdamW and the recommended Muon hybrid recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass
class MuonHybridOptimizer:
    """Pair Muon for Transformer matrices with AdamW for all other parameters."""

    muon: torch.optim.Optimizer
    adamw: torch.optim.Optimizer

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def _hybrid_parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return Muon-eligible block matrices and AdamW parameters, respectively."""
    muon_parameters: list[nn.Parameter] = []
    adamw_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("layers.") and parameter.ndim >= 2:
            muon_parameters.append(parameter)
        else:
            adamw_parameters.append(parameter)
    if not muon_parameters or not adamw_parameters:
        raise ValueError("Muon hybrid requires both Transformer matrices and auxiliary AdamW parameters")
    return muon_parameters, adamw_parameters


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer | MuonHybridOptimizer:
    """Create either all-AdamW or Muon-with-auxiliary-AdamW from YAML settings."""
    name = config["name"]
    if name == "adamw":
        settings = config["adamw"]
        return torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            betas=tuple(settings["betas"]),
            eps=settings["eps"],
            weight_decay=settings["weight_decay"],
        )
    if name == "muon_hybrid":
        settings = config["muon_hybrid"]
        muon_parameters, adamw_parameters = _hybrid_parameter_groups(model)
        muon = torch.optim.Muon(
            muon_parameters,
            lr=settings["muon_learning_rate"],
            momentum=settings["momentum"],
            nesterov=settings["nesterov"],
            ns_steps=settings["ns_steps"],
            weight_decay=settings["weight_decay"],
        )
        adamw = torch.optim.AdamW(
            adamw_parameters,
            lr=settings["auxiliary_adamw_learning_rate"],
            betas=tuple(settings["auxiliary_adamw_betas"]),
            eps=settings["auxiliary_adamw_eps"],
            weight_decay=settings["auxiliary_adamw_weight_decay"],
        )
        return MuonHybridOptimizer(muon=muon, adamw=adamw)
    raise ValueError(f"Unsupported optimizer: {name}")
