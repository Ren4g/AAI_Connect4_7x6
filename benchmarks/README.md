# Benchmark data

This directory contains replayable game experiments and their summaries.

- `round_robin/`: full agent matrix, paired fair-filter comparison, and fixed-depth regression replay.
- `head_to_head/`: focused matchups such as AlphaBeta versus MCTS and AlphaBeta depth 6 versus Heuristic.
- `ablation/`: paired multi-seed search-configuration comparison.

Each `.games.jsonl` file includes the run metadata and game actions needed for
replay. Keep summaries (`.json`), metadata (`.meta.json`), and replay logs
together. The root `results/` directory is reserved for solver evaluation,
training reports/labels, and CNN analysis artifacts.

Older pilot runs, unpaired duplicates, and pre-filter experiment variants that
were superseded by paired results were removed. The benchmark files retained
here are the ones used by the current report and solver-labeled training.
