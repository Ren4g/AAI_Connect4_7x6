# Phase C V2 protocol amendment (post-diagnostic, prospective)

Date: 2026-11-21, 12:33 Asia/Ho_Chi_Minh. This amendment was written after Phase A diagnostics and before the revised Phase C runs. It changes the planned Phase C settings for two new runs only. It does not revise or reinterpret any completed experiment. It is explicitly a post-diagnostic amendment, not part of the original preregistration.

## Motivation and question

Historical AZ PUCT-100 scored 66.5% exact-optimal on clean validation, while the initial supervised CNN under PUCT-100 scored 76.7%; the paired difference (AZ minus initial CNN) was -10.2 percentage points, 95% CI [-16.5, -4.0]. The revised experiment asks whether a fixed 400-simulation self-play run yields different validation learning curves from its own initialization, and how those curves compare between supervised warm-start and random initialization.

## Runs and fixed configuration

| Run | Run ID | Seed | Initialization | GPU cap |
|---|---|---:|---|---:|
| Warm-start | `az_v2_warm_seed20261121` | 20261121 | Best Phase B CNN v3 checkpoint chosen only on validation | 60 minutes |
| From-scratch | `az_v2_scratch_seed20261122` | 20261122 | Random model weights | 60 minutes |

The total Phase C GPU cap is 120 minutes. Both runs use 40 iterations, 32 self-play games per iteration, 16 parallel games, 400 simulations per move, 200 training updates per iteration, and batch size 128. Save iteration 0 and snapshots every 5 iterations through iteration 40. A resumable segment may use at most 30 minutes and must resume with the same run ID and config. Run the jobs sequentially.

The notebook maintains `artifacts/az_v2_phase_c_budget.json` on Drive. It reserves the segment allowance before launch and reconciles process elapsed time after normal completion. If the runtime dies before reconciliation, the reservation remains charged; do not reset or delete this ledger. The manifest and artifact ZIP include a copy whose SHA256 is added to the manifest. Archive snapshot files recursively.

## Analysis and limits

Plot iteration 0/5/.../40 on the validation set using PUCT with 400 simulations. Metrics are exact-optimal action rate and WDL preservation with 95% source-game-cluster bootstrap intervals; exclude, and report, source games present in the corresponding replay buffer. Compare each run's snapshots to its own iteration-0 checkpoint. After validation analysis is fixed, evaluate the iteration-40 checkpoints once on the frozen test; do not select a run or iteration using test results.

There is one seed per initialization. These runs support a controlled case comparison only; they do not satisfy the three-seed requirement for a general claim that one training method is superior. Report incomplete runs as incomplete. Stop at 40 iterations, 60 minutes per run, or 120 minutes total, whichever comes first. Do not extend the budget.
