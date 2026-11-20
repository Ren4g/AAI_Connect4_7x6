"""Paired-opening arena and predeclared promotion gate for two PUCT checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connect4.agents import PUCTAgent
from connect4.evaluation_data import sha256_file
from connect4.experiments import round_robin, save_game_logs, summarize_pair


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"}
               for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def promotion_decision(summary: dict, minimum_openings: int = 100) -> bool:
    interval = summary.get("score_bootstrap_95")
    return bool(
        summary["unique_openings"] >= minimum_openings
        and summary["score"] > 0.5
        and interval is not None and interval[0] > 0.5
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--incumbent", required=True, type=Path)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--incumbent-label", default="incumbent")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--games", type=int, default=200,
                        help="Paired games; gives 100 common openings by default")
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-openings", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-log", type=Path, required=True)
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.simulations < 1:
        parser.error("games must be positive/even and simulations positive")
    if args.candidate_label == args.incumbent_label:
        parser.error("candidate and incumbent labels must differ")
    minimum_openings = 2 if args.smoke else args.min_openings
    if not args.smoke and args.games // 2 < minimum_openings:
        parser.error("games do not provide the declared minimum paired openings")
    if args.output.exists() or args.games_log.exists():
        raise FileExistsError("arena outputs are immutable; choose new paths")

    candidate = PUCTAgent.from_checkpoint(args.candidate, args.simulations, device=args.device)
    incumbent = PUCTAgent.from_checkpoint(args.incumbent, args.simulations, device=args.device)
    names = {args.candidate_label: candidate, args.incumbent_label: incumbent}
    diagnostics: dict[str, int] = {}
    games = round_robin(
        names, games_per_pair=args.games, seed=args.seed,
        opening_plies=args.opening_plies, apply_fair_filter=True,
        opening_diagnostics=diagnostics, paired_openings=True,
    )
    summary = summarize_pair(args.candidate_label, args.incumbent_label,
                             games[(args.candidate_label, args.incumbent_label)])
    promote = promotion_decision(summary, minimum_openings)
    result = {
        "run_id": f"arena_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.seed}",
        "phase": "checkpoint_arena", "seed": args.seed,
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "source_code_commit": subprocess.run(["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip() or None,
        "source_code_sha256": source_tree_sha256(Path.cwd()),
        "python": platform.python_version(),
        "candidate": {"label": args.candidate_label, "path": str(args.candidate),
                      "sha256": sha256_file(args.candidate)},
        "incumbent": {"label": args.incumbent_label, "path": str(args.incumbent),
                      "sha256": sha256_file(args.incumbent)},
        "configuration": {"simulations_per_move": args.simulations,
                          "games": args.games, "opening_plies": args.opening_plies,
                          "minimum_openings": minimum_openings, "paired_openings": True,
                          "fair_opening_filter": True, "smoke": args.smoke},
        "opening_filter": diagnostics, "summary": summary,
        "promotion_gate": "score > 0.50 and paired-opening bootstrap lower 95% CI > 0.50",
        "promoted": promote and not args.smoke,
        "bootstrap_seed": 20261119,
    }
    if promote and not args.smoke and args.promote_to:
        if args.promote_to.exists():
            raise FileExistsError(f"refusing to overwrite promoted checkpoint {args.promote_to}")
        args.promote_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate, args.promote_to)
        result["promoted_checkpoint"] = str(args.promote_to)
    elif promote and not args.smoke:
        result["promoted"] = False
        result["promotion_note"] = "gate passed, but --promote-to was not supplied"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_game_logs(games, args.games_log, metadata=result)
    if not args.smoke:
        with Path("docs/LOG.md").open("a", encoding="utf-8") as log:
            log.write(f"\n### {datetime.now().astimezone().date()} — Checkpoint arena\n\n")
            log.write(f"- Result: `{args.output}`; replayable games: `{args.games_log}`; "
                      f"score={summary['score']:.4f}, games={summary['games']}, "
                      f"openings={summary['unique_openings']}, promoted={result['promoted']}.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
