from __future__ import annotations

from typing import Any


class ConnectFourCNN:
    """Small policy/value CNN factory; PyTorch is imported only when used."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        try:
            import torch.nn as nn
        except ImportError as error:
            raise RuntimeError("Install PyTorch to use ConnectFourCNN") from error

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(2, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                self.policy = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 7, 7))
                self.value = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 7, 1), nn.Tanh())

            def forward(self, inputs: Any) -> tuple[Any, Any]:
                features = self.features(inputs)
                return self.policy(features), self.value(features).squeeze(-1)

        return _Network(*args, **kwargs)
