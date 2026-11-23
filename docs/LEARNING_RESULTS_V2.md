# V2 Results: Learning Agents for Connect Four

> This report separates local diagnostics from Colab training. New metrics use source-game validation clusters. Frozen-test figures are historical references only and were not used to select V2 configurations.

## Progress summary

- Phase A completed locally. The CNN v2 dataset contains 1,500 positions: 1,284 train and 216 validation.
- The three historical CNN runs used batch size 64, up to 30 epochs, and patience 6; completed updates were 609, 630, 630. The train split yields 21 batches per epoch; the first run stopped early.
- Pons exact produced 30,000 labels for dataset v3. The AlphaBeta-3 policy control uses the same positions and retains the Pons value targets.
- No CNN v3 or AlphaZero V2 run has been executed on Colab. No new training outcome is claimed.

## 1. CNN v2 data audit

Dataset v2 contains 1,284 train positions from 1,023 source games and 216 validation positions from 181 games. Labels use Pons exact scores, softmax policy targets at temperature 2.0, and the sign of the best solver score for value. The manifest reports zero frozen-test overlap.
With batch size 64, the train split yields 21 updates per epoch. Historical test exact-optimal rates (n=1,500 each) were: CNN-CNN_v2_s20261120-policy: 70.7%, CI [68.3%, 72.9%]; CNN-CNN_v2_s20261121-policy: 70.9%, CI [68.6%, 73.2%]; CNN-CNN_v2_s20261122-policy: 71.2%, CI [68.8%, 73.6%]. These test results are historical and were not used to select V2 runs.

## 2. Policy, PUCT, and safety-filter diagnostics

The clean validation subset contains 215 positions in 180 source-game clusters. One validation state was present in the AZ replay and was excluded from AZ comparisons. No remaining row overlaps the frozen test set.

| Method | Exact-optimal action | 95% source-game cluster bootstrap CI |
|---|---:|---:|
| cnn_policy_raw | 119/215 (55.3%) | [48.4%, 62.0%] |
| cnn_policy_safety_filter | 158/215 (73.5%) | [67.5%, 79.4%] |
| cnn_puct_100 | 165/215 (76.7%) | [70.9%, 82.4%] |
| az_puct_100 | 143/215 (66.5%) | [59.9%, 72.9%] |

On identical positions, the CNN safety filter improved exact-optimal rate by 18.1% (paired CI [13.0%, 23.3%]). PUCT-100 with the initial CNN exceeded raw policy by 21.4%. Final AZ PUCT was lower than initial-CNN PUCT by 10.2 percentage points (paired AZ-minus-initial CI [-16.5%, -4.0%]). These results describe this validation sample, not a multi-seed method claim.
A separate paired PUCT filter run on the same 215 validation positions found filter gains of 2.3% for initial CNN (CI [0.5%, 4.5%]) and 7.9% for final AZ (CI [4.5%, 11.7%]). PUCT filtered rates were 79.1% for initial CNN and 74.4% for AZ. Safety filtering therefore contributes to PUCT results as well as policy-only results.

## 3. Value head

After excluding the same AZ replay state, validation contains 215 positions. Initial CNN W/D/L accuracy was 70.7% [64.5%, 76.7%]; final AZ was 69.8% [63.4%, 76.1%]. The paired AZ-minus-initial accuracy difference was -0.9 points, CI [-8.0%, 6.1%]; it is not distinguished. AZ value MSE increased by 0.176 (paired CI 0.019 to 0.333).
The majority-class baseline reached 57.7% W/D/L accuracy (CI [51.4%, 64.1%]) and MSE 1.540. Paired clustered intervals show the initial CNN and final AZ accuracy deltas over this baseline as +0.130 (CI [0.04608294930875576, 0.2132701421800948]) and +0.121 (CI [0.03686635944700461, 0.20276497695852536]).

## 4. Level-6 RL diagnostic

The historical RL run completed 10,000 episodes against Random only. Its mean logged reward was 0.676 over those episodes. In the historical paired-opening round-robin, Random versus RL-Policy had n=100, score 0.020, bootstrap CI [0.0%, 5.0%]; the opponents and opening distribution differ, so training reward is not a general playing-strength measure.
Episode-0 was reconstructed from seed 20261120 using local PyTorch because the original initial checkpoint was not saved. The local PyTorch version differs from Colab, so this is a seeded random-initialization proxy. Exact-optimal rates raw/filtered were 36.0%/63.6% for the proxy and 42.5%/66.4% after training. The paired trained-filter minus episode-0-filter delta was 2.8%, CI [-3.3%, 8.9%]; this does not distinguish learning from initialization.

## 5. Solver cost and v3 data

The prior 50-position exact probe took 44.4 seconds. Mean cost was 2.604 seconds/opening, 0.0055 seconds/middle, and 0.000175 seconds/endgame. Weak mode on the same positions took 20.5 seconds. No `7x6.book` exists under `solver/`. The strong ply-dependent search cost explains the observed runtime better than the Python wrapper overhead.
Dataset v3 generated 30,000 exact labels in 2172.5 seconds with 12 workers (13.81 positions/second aggregate). It is balanced at 10,000 positions per stage, has zero frozen-test overlap, and contains 20,517 source-game clusters. Selection rejected 79 frozen-test states and 3194 internal duplicates/mirrors. The matched AlphaBeta-3 teacher policy package contains 30,000 positions and took 54.9 seconds to prepare on 12 workers.

## 6. Limits and next steps

- The historical AlphaZero run has one seed and no iteration checkpoints. Its training loss does not establish stronger play.
- Source-game cluster bootstrap accounts for within-game dependence but does not replace independent training seeds.
- Tactical safety filters contribute materially for policy and PUCT agents and must remain separately reported from raw network output.
- CNN v3 and controlled AZ V2 notebooks are prepared but have not been run on Colab. The total Colab GPU cap remains 180 minutes. RL is not scheduled because validation did not distinguish trained policy from the episode-0 proxy with the filter applied.

## Source JSON

All metrics above are rendered from these JSON files:

- `results/diagnostic_v2_validation_clean_20261121_seed20261121.json`
- `results/diagnostic_v2_puct_filter_validation_20261121_seed20261121.json`
- `results/diagnostic_v2_value_ci_supplement_20261121_seed20261121.json`
- `results/diagnostic_v2_value_paired_baselines_20261121_seed20261121.json`
- `results/diagnostic_v2_rl_episode0_validation_20261121_seed20261121.json`
- `results/solver_probe_20261120_seed2701_50.json`
- `results/solver_weak_probe_20261121_seed2701_50.json`
- `colab/data/solverlabels_20261121_seed20261121_v3_30k/manifest.json`
- `colab/data/solverlabels_20261121_seed20261121_ab3_30k/manifest.json`

Render: `python -m scripts.render_learning_results_v2 --output docs/LEARNING_RESULTS_V2.md`.
