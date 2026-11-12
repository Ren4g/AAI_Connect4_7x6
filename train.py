from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def board_tensor(board: tuple[tuple[int, ...], ...], player: int) -> Any:
    import torch

    opponent = 1 if player == 2 else 2
    return torch.tensor(
        [
            [[float(cell == player) for cell in row] for row in board],
            [[float(cell == opponent) for cell in row] for row in board],
        ],
        dtype=torch.float32,
    )


def train(
    games: int = 100,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    checkpoint: str = "checkpoints/connect4_cnn.pt",
) -> list[dict[str, float]]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError(
            "Training requires PyTorch. Install it with: python -m pip install torch"
        ) from error

    from connect4.model import ConnectFourCNN
    from connect4.training import build_training_records

    if games < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("games, epochs and batch_size must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    records = build_training_records(games, seed=seed)
    random.Random(seed).shuffle(records)
    split = max(1, int(len(records) * 0.8))
    train_records = records[:split]
    validation_records = records[split:] or records[-1:]

    def tensors(items: list[Any]) -> tuple[Any, Any, Any]:
        inputs = torch.stack([board_tensor(item.board, item.player) for item in items])
        actions = torch.tensor([item.action for item in items], dtype=torch.long)
        values = torch.tensor([item.value for item in items], dtype=torch.float32)
        return inputs, actions, values

    train_inputs, train_actions, train_values = tensors(train_records)
    valid_inputs, valid_actions, valid_values = tensors(validation_records)
    loader = DataLoader(
        TensorDataset(train_inputs, train_actions, train_values),
        batch_size=batch_size,
        shuffle=True,
    )
    model = ConnectFourCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, actions, values in loader:
            optimizer.zero_grad()
            policy_logits, predicted_values = model(inputs)
            loss = policy_loss(policy_logits, actions) + value_loss(predicted_values, values)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
        model.eval()
        with torch.no_grad():
            valid_policy, valid_value = model(valid_inputs)
            valid_loss = policy_loss(valid_policy, valid_actions) + value_loss(valid_value, valid_values)
            accuracy = (valid_policy.argmax(dim=1) == valid_actions).float().mean().item()
        metrics = {
            "epoch": float(epoch),
            "train_loss": total_loss / len(train_records),
            "validation_loss": float(valid_loss.item()),
            "validation_policy_accuracy": float(accuracy),
        }
        history.append(metrics)
        print(json.dumps(metrics))

    output = Path(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "seed": seed, "games": games, "history": history},
        output,
    )
    print(f"Saved checkpoint to {output}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Connect Four policy/value CNN")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/connect4_cnn.pt")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
