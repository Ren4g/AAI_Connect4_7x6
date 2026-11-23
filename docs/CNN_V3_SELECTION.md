# CNN v3 validation selection (2026-11-22)

Configuration comparison uses validation exact-optimal top-1, averaged over the three registered seeds. Checkpoint within a run is the validation-loss-selected `checkpoint_best.pt`. The frozen solver test was not used in this selection.

| Cấu hình | Seed 1/2/3 exact-optimal | Mean ± SD | Mean validation loss |
|---|---:|---:|---:|
| Pons soft · 100% | 79.82% / 80.40% / 80.04% | 80.09% ± 0.29 pp | 1.4678 |
| Pons hard · 100% | 75.52% / 77.95% / 77.33% | 76.93% ± 1.26 pp | 1.0043 |
| AB3 teacher · 100% | 73.02% / 73.27% / 73.49% | 73.26% ± 0.23 pp | 1.0769 |
| Pons soft · 30% | 72.43% / 72.94% / 72.30% | 72.55% ± 0.34 pp | 1.6301 |
| Pons soft · 10% | 63.40% / 63.47% / 63.16% | 63.34% ± 0.16 pp | 1.8076 |

Selected configuration: **Pons soft · 100%**.
Warm-start checkpoint: `cnn_v3_30k_seed20261122/checkpoint_best.pt` (seed 20261122, epoch 30); validation exact-optimal 80.40% on 4530 positions, validation loss 1.4617.
Checkpoint copied to `checkpoints/cnn_v3_30k_best.pt`; SHA256 `3d86468ab7586ef502e8b1a894909b722506192b88fcc8a9937b814fb96ff35b`.
The three seeds support a descriptive validation comparison only; do not treat their SD as a cluster-bootstrap CI.
