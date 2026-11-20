from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from connect4.engine import COLS, ConnectFour, GameResult
from connect4.solver import PonsSolver


PHASES = {"opening": (7, 10), "middle": (11, 24), "endgame": (25, 41)}


def canonical_state(moves: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], int]:
    state = ConnectFour()
    for move in moves:
        state.step(move)
    board = tuple(tuple(row) for row in state.board)
    mirror = tuple(tuple(reversed(row)) for row in board)
    return min(board, mirror), int(state.current_player)


def sample_log(path: str, per_stage: int, seed: int):
    rng = random.Random(seed)
    candidates: dict[str, list[tuple[str, tuple[int, ...]]]] = {
        phase: [] for phase in PHASES
    }
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            record = json.loads(line)
            if record.get("record_type") != "game":
                continue
            actions = record["actions"]
            game_id = f"{record['pair'][0]}|{record['pair'][1]}|{record['game_index']}"
            for phase, (low, high) in PHASES.items():
                high = min(high, len(actions) - 1)
                if high >= low:
                    ply = rng.randint(low, high)
                    candidates[phase].append((game_id, tuple(actions[:ply])))

    sampled = {}
    metadata = {}
    for phase, rows in candidates.items():
        rng.shuffle(rows)
        seen = set()
        unique = []
        for game_id, moves in rows:
            key = canonical_state(moves)
            if key in seen:
                continue
            seen.add(key)
            unique.append((game_id, moves))
        if len(unique) < per_stage:
            raise ValueError(f"Only {len(unique)} unique {phase} positions in {path}; requested {per_stage}")
        sampled[phase] = unique[:per_stage]
        metadata[phase] = {"candidates": len(rows), "unique_mirror_classes": len(unique),
                           "selected": len(sampled[phase])}
    return sampled, metadata


def split_train_val(train_sample, validation_fraction: float, seed: int):
    all_rows = [row for rows in train_sample.values() for row in rows]
    game_ids = sorted({game_id for game_id, _ in all_rows})
    random.Random(seed).shuffle(game_ids)
    split_at = max(1, min(len(game_ids) - 1, round(len(game_ids) * (1 - validation_fraction))))
    train_games, validation_games = set(game_ids[:split_at]), set(game_ids[split_at:])
    train_rows = [row for row in all_rows if row[0] in train_games]
    validation_rows = [row for row in all_rows if row[0] in validation_games]

    train_keys = {canonical_state(moves) for _, moves in train_rows}
    validation_rows = [row for row in validation_rows if canonical_state(row[1]) not in train_keys]
    validation_keys = {canonical_state(moves) for _, moves in validation_rows}
    test_keys = train_keys | validation_keys
    return train_rows, validation_rows, train_keys, validation_keys, test_keys


def solver_labels(rows, solver: PonsSolver, temperature: float):
    labeled = []
    for game_id, moves in rows:
        state = ConnectFour()
        for move in moves:
            state.step(move)
        scores = solver.scores(moves)
        legal = state.legal_actions()
        best = max(scores[column] for column in legal)
        weights = [0.0] * COLS
        max_logit = max(scores[column] for column in legal) / temperature
        exp_weights = {column: math.exp(scores[column] / temperature - max_logit) for column in legal}
        normalizer = sum(exp_weights.values())
        for column, weight in exp_weights.items():
            weights[column] = weight / normalizer
        value = float((best > 0) - (best < 0))
        labeled.append({"source_game": game_id, "moves": list(moves),
                        "policy": weights, "value": value, "best_score": best,
                        "optimal_actions": [column for column in legal if scores[column] == best]})
    return labeled


def mirror_record(record):
    return {
        **record,
        "moves": [COLS - 1 - column for column in record["moves"]],
        "policy": list(reversed(record["policy"])),
        "optimal_actions": [COLS - 1 - column for column in record["optimal_actions"]],
    }


def run(args):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise RuntimeError("Training solver labels requires PyTorch") from error
    from connect4.encoding import encode_planes
    from connect4.model import ConnectFourCNN

    started = time.perf_counter()
    train_sample, train_sample_meta = sample_log(args.train_log, args.per_stage, args.seed)
    test_sample, test_sample_meta = sample_log(args.test_log, args.per_stage, args.seed + 1)
    train_rows, validation_rows, train_keys, validation_keys, _ = split_train_val(
        train_sample, args.validation_fraction, args.seed
    )
    test_rows = [row for rows in test_sample.values() for row in rows]
    test_rows = [row for row in test_rows if canonical_state(row[1]) not in train_keys | validation_keys]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Game/source split or position de-duplication produced an empty split")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cache_path = Path(args.labels)
    cache_signature = {
        "train_log": args.train_log, "test_log": args.test_log,
        "per_stage": args.per_stage, "seed": args.seed,
        "temperature": args.temperature,
    }
    if cache_path.exists() and not args.regenerate_labels:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("signature") != cache_signature:
            raise ValueError("Solver-label cache does not match this run; choose another --labels or pass --regenerate-labels")
        labeled = cache["records"]
        lookup = {(row["source_game"], tuple(row["moves"])): row for row in labeled}
        def get_labels(rows):
            return [lookup[(game, moves)] for game, moves in rows]
        train_records = get_labels(train_rows)
        validation_records = get_labels(validation_rows)
        test_records = get_labels(test_rows)
    else:
        with PonsSolver(args.solver, timeout_seconds=args.solver_timeout) as solver:
            unique_rows = {}
            for row in train_rows + validation_rows + test_rows:
                unique_rows[(row[0], row[1])] = row
            labeled = solver_labels(list(unique_rows.values()), solver, args.temperature)
        lookup = {(row["source_game"], tuple(row["moves"])): row for row in labeled}
        def get_labels(rows):
            return [lookup[(game, moves)] for game, moves in rows]
        train_records = get_labels(train_rows)
        validation_records = get_labels(validation_rows)
        test_records = get_labels(test_rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"signature": cache_signature, "records": labeled}), encoding="utf-8")

    # Reflection is augmentation for train only; all validation/test mirrors
    # remain excluded by canonical_state above.
    train_records += [mirror_record(row) for row in train_records]

    class PositionDataset(Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            row = self.records[index]
            state = ConnectFour()
            for action in row["moves"]:
                state.step(action)
            inputs = torch.tensor(encode_planes(state.board, int(state.current_player)), dtype=torch.float32)
            optimal = [column in row["optimal_actions"] for column in range(COLS)]
            return (inputs, torch.tensor(row["policy"], dtype=torch.float32),
                    torch.tensor(row["value"], dtype=torch.float32),
                    torch.tensor(optimal, dtype=torch.bool))

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(PositionDataset(train_records), batch_size=args.batch_size,
                            shuffle=True, generator=generator, pin_memory=device.type == "cuda"),
        "validation": DataLoader(PositionDataset(validation_records), batch_size=args.batch_size,
                                 shuffle=False, pin_memory=device.type == "cuda"),
        "test": DataLoader(PositionDataset(test_records), batch_size=args.batch_size,
                            shuffle=False, pin_memory=device.type == "cuda"),
    }
    model = ConnectFourCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    value_loss = nn.MSELoss()
    history = []
    best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0

    def evaluate(loader, training=False):
        model.train(training)
        total_loss = policy_total = value_total = 0.0
        optimal_correct = positions = 0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for inputs, targets, values, optimal_mask in loader:
                inputs, targets, values = inputs.to(device), targets.to(device), values.to(device)
                optimal_mask = optimal_mask.to(device)
                logits, predicted_values = model(inputs)
                log_probs = torch.log_softmax(logits, dim=1)
                policy_batch = -(targets * log_probs).sum(dim=1).mean()
                value_batch = value_loss(predicted_values.view(-1), values)
                loss = policy_batch + value_batch
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                count = len(inputs)
                total_loss += float(loss.detach()) * count
                policy_total += float(policy_batch.detach()) * count
                value_total += float(value_batch.detach()) * count
                pred = logits.argmax(dim=1).tolist()
                for index, action in enumerate(pred):
                    if bool(optimal_mask[index, action]):
                        optimal_correct += 1
                positions += count
        return {"loss": total_loss / positions, "policy_loss": policy_total / positions,
                "value_mse": value_total / positions, "optimal_action_accuracy": optimal_correct / positions}

    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = evaluate(loaders["train"], training=True)
        validation_metrics = evaluate(loaders["validation"])
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_val:
            best_val = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(json.dumps(history[-1]), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(loaders["test"])
    training_seconds = time.perf_counter() - started
    report = {
        "status": "complete", "seed": args.seed, "device": str(device),
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "train_log": args.train_log, "test_log": args.test_log,
        "train_sampling": train_sample_meta, "test_sampling": test_sample_meta,
        "split_sizes_after_mirror": {"train": len(train_records),
                                      "validation": len(validation_records),
                                      "test": len(test_records)},
        "train_games": len({g for g, _ in train_rows}),
        "validation_games": len({g for g, _ in validation_rows}),
        "test_games": len({g for g, _ in test_rows}),
        "position_keys_disjoint": True,
        "temperature": args.temperature, "best_epoch": best_epoch,
        "history": history, "test_metrics": test_metrics,
        "training_seconds": training_seconds,
    }
    torch.save({"model_state_dict": model.state_dict(), **report}, output)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved solver-labeled checkpoint to {output}")
    print(f"Saved report to {args.report}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train policy/value CNN on exact solver labels")
    parser.add_argument("--train-log", default="benchmarks/round_robin/round_robin_full_seed20261119.games.jsonl")
    parser.add_argument("--test-log", default="benchmarks/round_robin/round_robin_filtered_paired_seed20261124.games.jsonl")
    parser.add_argument("--solver", default="solver/connect4/c4solver.exe")
    parser.add_argument("--solver-timeout", type=float, default=300)
    parser.add_argument("--per-stage", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--labels", default="results/solver_labels_seed20261201.json")
    parser.add_argument("--regenerate-labels", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/connect4_cnn_solver_seed20261201.pt")
    parser.add_argument("--report", default="results/training_solver_seed20261201.json")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.per_stage < 1 or args.temperature <= 0:
        parser.error("validation fraction, per-stage, and temperature must be positive and in range")
    run(args)


if __name__ == "__main__":
    main()
