# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Compute FLOPS of normalizing flow model."""

import torch
from fvcore.nn import FlopCountAnalysis

from infqm.normalizing_flow.train import RealNVP


def main() -> None:
    """Evaluate the FLOPs of the RealNVP model."""
    checkpoint = torch.load(
        "network_logs/train_20251029_163853/normalizing_flow_model.pth",
        map_location="cpu",
    )

    model = RealNVP(
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        mode=checkpoint["mode"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_tensor = torch.zeros(12).reshape(1, -1)
    FlopCountAnalysis(model, input_tensor)


if __name__ == "__main__":
    main()
