from __future__ import annotations

import argparse
import json
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
    epsilon: float = 0.05,
    engine: str = "grid",
    device: str = "auto",
) -> list[dict[str, float]]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise RuntimeError(
            "Training requires PyTorch. Install it with: python -m pip install torch"
        ) from error

    from connect4.model import ConnectFourCNN
    from connect4.training import split_training_records
    from connect4.engine import ConnectFour
    from connect4.engine_bit import ConnectFour as BitboardConnectFour

    if games < 3 or epochs < 1 or batch_size < 1:
        raise ValueError("games must be at least 3; epochs and batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if engine not in {"grid", "bitboard"}:
        raise ValueError("engine must be 'grid' or 'bitboard'")
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.cuda.manual_seed_all(seed) if target_device.type == "cuda" else None
    torch.use_deterministic_algorithms(True, warn_only=True)

    engine_factory = BitboardConnectFour if engine == "bitboard" else ConnectFour
    splits = split_training_records(
        games, seed=seed, epsilon=epsilon, engine_factory=engine_factory
    )

    class PositionDataset(Dataset):
        def __init__(self, records: list[Any]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
            record = self.records[index]
            return (
                board_tensor(record.board, record.player),
                torch.tensor(record.action, dtype=torch.long),
                torch.tensor(record.value, dtype=torch.float32),
            )

    train_records = splits["train"]
    validation_records = splits["validation"]
    test_records = splits["test"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PositionDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=target_device.type == "cuda",
    )
    validation_loader = DataLoader(
        PositionDataset(validation_records), batch_size=batch_size,
        shuffle=False, pin_memory=target_device.type == "cuda",
    )
    test_loader = DataLoader(
        PositionDataset(test_records), batch_size=batch_size,
        shuffle=False, pin_memory=target_device.type == "cuda",
    )

    model = ConnectFourCNN().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, actions, values in loader:
            inputs = inputs.to(target_device, non_blocking=True)
            actions = actions.to(target_device, non_blocking=True)
            values = values.to(target_device, non_blocking=True)
            optimizer.zero_grad()
            policy_logits, predicted_values = model(inputs)
            loss = policy_loss(policy_logits, actions) + value_loss(predicted_values, values)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
        model.eval()
        validation_metrics = evaluate_loader(
            model, validation_loader, target_device, policy_loss, value_loss
        )
        metrics = {
            "epoch": float(epoch),
            "train_loss": total_loss / len(train_records),
            "validation_loss": validation_metrics["loss"],
            "validation_policy_accuracy": validation_metrics["policy_accuracy"],
        }
        history.append(metrics)
        print(json.dumps(metrics))

    output = Path(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = evaluate_loader(model, test_loader, target_device, policy_loss, value_loss)
    torch.save({
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "games": games,
        "epsilon": epsilon,
        "engine": engine,
        "device": str(target_device),
        "split_sizes": {name: len(records) for name, records in splits.items()},
        "history": history,
        "test_metrics": test_metrics,
    }, output)
    print(f"Saved checkpoint to {output}")
    print(json.dumps({"test": test_metrics, "split_sizes": {name: len(items) for name, items in splits.items()}}))
    return history


def evaluate_loader(model: Any, loader: Any, device: Any, policy_loss: Any, value_loss: Any) -> dict[str, float]:
    import torch

    model.eval()
    total_loss = 0.0
    correct = total = 0
    with torch.inference_mode():
        for inputs, actions, values in loader:
            inputs = inputs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            policy_logits, predicted_values = model(inputs)
            count = len(inputs)
            batch_loss = policy_loss(policy_logits, actions) + value_loss(predicted_values, values)
            total_loss += float(batch_loss.item()) * count
            correct += int((policy_logits.argmax(dim=1) == actions).sum().item())
            total += count
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset split")
    return {"loss": total_loss / total, "policy_accuracy": correct / total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Connect Four policy/value CNN")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/connect4_cnn.pt")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--engine", choices=("grid", "bitboard"), default="grid")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
