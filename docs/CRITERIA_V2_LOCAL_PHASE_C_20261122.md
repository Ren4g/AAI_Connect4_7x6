# Phase C V2 — local execution amendment

**Registered:** 2026-11-22, before the local Phase C V2 runs.
**Reason:** the project owner observed low GPU utilization during Colab self-play and requested local execution on the available RTX 3050 laptop, using all available CPU threads, with both runs continued to completion. Existing Colab artifacts do not include GPU utilization traces, so CPU bottleneck is an operational observation, not a conclusion from the archived manifests.

## Scope of this amendment

This amendment changes the execution location, resource accounting, and interruption handling for the two not-yet-completed Phase C V2 runs. It does not modify or reinterpret any completed experiment. The original `docs/CRITERIA_V2.md` and `docs/CRITERIA_V2_POSTHOC.md` remain historical records. The Phase C research question, initializations, seeds, model, data, and algorithm remain fixed.

The owner explicitly requests both runs be continued until their registered iteration target is reached. The previous Colab GPU-minute cap and Drive budget ledger therefore do not apply to these local runs. There is no fixed local wall-time cap in this amendment. Actual elapsed local time, pauses, resumes, and completed work must be recorded; no elapsed-time claim may be compared with Colab as an agent-strength metric.

## Locked runs and configuration

| Run | Run ID | Seed | Initialization |
|---|---|---:|---|
| Warm-start | `az_v2_warm_seed20261121` | 20261121 | Validation-selected `checkpoints/cnn_v3_30k_best.pt` |
| From-scratch | `az_v2_scratch_seed20261122` | 20261122 | Random weights |

Both runs retain the registered configuration: 40 iterations; 32 games per iteration; 16 parallel game roots; 400 simulations per move; 200 training updates per iteration; batch size 128; snapshot iteration 0 and every 5 iterations through 40; replay capacity, learning rate, value weight, PUCT constant, Dirichlet settings, and temperature schedule as defined by the existing Phase C notebook and `connect4.selfplay` defaults. Warm and scratch run sequentially. Do not change hyperparameters when resuming.

Local environment recorded before execution: Windows, Python/PyTorch/CUDA versions and GPU from the active `.venv-gpu`, GPU model NVIDIA GeForce RTX 3050 Laptop GPU, and 12 logical CPUs (`os.cpu_count()`). Set PyTorch intra-op and BLAS/OpenMP thread limits to 12. Python PUCT tree traversal itself is serial; the run is permitted to use all available threads for operations that support intra-op parallelism, but do not claim 12-core scaling unless measured.

## Interruption and checkpoint policy

- Store each run under a new immutable directory `results/phase_c_local/<run_id>/`; do not overwrite Colab artifacts or prior results.
- Save `checkpoint_latest.pt` and its manifest at least every 120 seconds; also save registered iteration snapshots. Before launching a segment, record a reservation in `results/phase_c_local/phase_c_local_ledger.json`; reconcile elapsed time after a normal exit. If interrupted before reconciliation, leave the reservation charged conservatively.
- Resume a partial run only with its existing run ID, seed, initialization, dataset manifest, source tree, and exact configuration. Check the latest checkpoint and manifest before restarting. A checkpoint/config mismatch is a hard stop requiring diagnosis; never restart into the same result directory by deleting files.
- Use resumable segments of at most 15 minutes so that a terminal interruption does not discard more than the periodic-checkpoint interval plus the current segment's uncheckpointed work. The training checkpoint preserves active games, replay, optimizer, counters, and RNG state.
- A run is complete only at iteration 40 with a complete manifest and all scheduled snapshots. If a process stops unexpectedly, resume from the latest valid checkpoint; do not create a new seed or reset the run.

## Analysis and reporting order

1. After both runs complete, evaluate snapshots 0/5/.../40 on the same validation positions using PUCT-400. Exclude and report replay-source-game overlap by checkpoint; report exact-optimal and WDL-preserving rates with source-game-cluster bootstrap intervals. Compare each run with its own iteration-0 model.
2. Freeze the validation analysis before evaluating iteration-40 checkpoints once on the frozen test set. Do not use test results to choose an initialization or iteration.
3. Ingest/validate manifests and checkpoints, then run the updated local round-robin including CNN v3 and the two final Phase C agents. Keep the earlier round-robin as a historical baseline.
4. This remains a one-seed-per-initialization case comparison. It cannot support a general claim that warm-start or scratch is superior across seeds. Report incomplete or negative outcomes without changing this interpretation.
