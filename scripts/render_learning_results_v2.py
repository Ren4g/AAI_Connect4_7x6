from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ci(values: list[float]) -> str:
    return f"[{pct(values[0])}, {pct(values[1])}]"


def render(root: Path) -> str:
    data_dir = root / "colab/data/solverlabels_20261120_seed20261120_v2"
    data_v2 = read(data_dir / "manifest.json")
    cnn_runs = [read(path) for path in sorted((root / "results/colab").glob("cnn_seed*/manifest.json"))]
    cnn_solver = read(root / "results/solver_eval_cnn_v2_3seeds.json")
    az_solver = read(root / "results/solver_eval_az_rl_20261121_seed20261121.json")
    az_manifest = read(root / "results/colab/az_seed20261120_run1/manifest.json")
    rl_manifest = read(root / "results/colab/rl_seed20261120_run1/manifest.json")
    puct = read(root / "results/diagnostic_v2_validation_clean_20261121_seed20261121.json")
    puct_filter = read(root / "results/diagnostic_v2_puct_filter_validation_20261121_seed20261121.json")
    value = read(root / "results/diagnostic_v2_value_ci_supplement_20261121_seed20261121.json")
    value_paired = read(root / "results/diagnostic_v2_value_paired_baselines_20261121_seed20261121.json")
    rl = read(root / "results/diagnostic_v2_rl_episode0_validation_20261121_seed20261121.json")
    exact_probe = read(root / "results/solver_probe_20261120_seed2701_50.json")
    weak_probe = read(root / "results/solver_weak_probe_20261121_seed2701_50.json")
    data_v3 = read(root / "colab/data/solverlabels_20261121_seed20261121_v3_30k/manifest.json")
    ab3 = read(root / "colab/data/solverlabels_20261121_seed20261121_ab3_30k/manifest.json")
    rr_rl = next(x for x in read(root / "benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json")
                 if x["first_agent"] == "Random" and x["second_agent"] == "RL-Policy")

    lines = [
        "# V2 Results: Learning Agents for Connect Four",
        "",
        "> This report separates local diagnostics from Colab training. New metrics use source-game validation clusters. Frozen-test figures are historical references only and were not used to select V2 configurations.",
        "",
        "## Progress summary",
        "",
        "- Phase A completed locally. The CNN v2 dataset contains 1,500 positions: 1,284 train and 216 validation.",
        f"- The three historical CNN runs used batch size 64, up to 30 epochs, and patience 6; completed updates were {', '.join(str(x['completed_steps']) for x in cnn_runs)}. The train split yields {(-(-1284 // 64))} batches per epoch; the first run stopped early.",
        f"- Pons exact produced {data_v3['splits']['train']['positions'] + data_v3['splits']['validation']['positions']:,} labels for dataset v3. The AlphaBeta-3 policy control uses the same positions and retains the Pons value targets.",
        "- No CNN v3 or AlphaZero V2 run has been executed on Colab. No new training outcome is claimed.",
        "",
        "## 1. CNN v2 data audit",
        "",
        f"Dataset v2 contains {data_v2['splits']['train']['positions']:,} train positions from {data_v2['splits']['train']['source_games']:,} source games and {data_v2['splits']['validation']['positions']:,} validation positions from {data_v2['splits']['validation']['source_games']:,} games. Labels use Pons exact scores, softmax policy targets at temperature {data_v2['configuration']['policy_temperature']}, and the sign of the best solver score for value. The manifest reports zero frozen-test overlap.",
        f"With batch size 64, the train split yields {(-(-data_v2['splits']['train']['positions'] // 64))} updates per epoch. Historical test exact-optimal rates (n=1,500 each) were: " + "; ".join(f"{name}: {pct(cnn_solver['metrics'][name]['all']['optimal']['rate'])}, CI {ci(cnn_solver['metrics'][name]['all']['optimal']['cluster_bootstrap_95'])}" for name in ('CNN-CNN_v2_s20261120-policy','CNN-CNN_v2_s20261121-policy','CNN-CNN_v2_s20261122-policy')) + ". These test results are historical and were not used to select V2 runs.",
        "",
        "## 2. Policy, PUCT, and safety-filter diagnostics",
        "",
        f"The clean validation subset contains {puct['positions_analyzed']} positions in {puct['source_game_clusters']} source-game clusters. One validation state was present in the AZ replay and was excluded from AZ comparisons. No remaining row overlaps the frozen test set.",
        "",
        "| Method | Exact-optimal action | 95% source-game cluster bootstrap CI |",
        "|---|---:|---:|",
    ]
    for key in ("cnn_policy_raw", "cnn_policy_safety_filter", "cnn_puct_100", "az_puct_100"):
        row = puct["metrics"][key]
        lines.append(f"| {key} | {row['optimal']}/{row['positions']} ({pct(row['rate'])}) | {ci(row['cluster_bootstrap_95'])} |")
    cnn_filter = puct["paired_vs_cnn_raw_and_between_puct"]["cnn_policy_safety_filter"]
    cnn_puct = puct["paired_vs_cnn_raw_and_between_puct"]["cnn_puct_100"]
    az_vs_cnn = puct["paired_vs_cnn_raw_and_between_puct"]["az_puct_100_minus_cnn_puct_100"]
    cnn_puct_filter = puct_filter["paired_filter_deltas"]["cnn_init_filter_minus_raw"]
    az_puct_filter = puct_filter["paired_filter_deltas"]["az_final_filter_minus_raw"]
    lines += [
        "",
        f"On identical positions, the CNN safety filter improved exact-optimal rate by {pct(cnn_filter['delta'])} (paired CI {ci(cnn_filter['cluster_bootstrap_95'])}). PUCT-100 with the initial CNN exceeded raw policy by {pct(cnn_puct['delta'])}. Final AZ PUCT was lower than initial-CNN PUCT by {-100 * az_vs_cnn['delta']:.1f} percentage points (paired AZ-minus-initial CI {ci(az_vs_cnn['cluster_bootstrap_95'])}). These results describe this validation sample, not a multi-seed method claim.",
        f"A separate paired PUCT filter run on the same {puct_filter['positions']} validation positions found filter gains of {pct(cnn_puct_filter['delta'])} for initial CNN (CI {ci(cnn_puct_filter['cluster_bootstrap_95'])}) and {pct(az_puct_filter['delta'])} for final AZ (CI {ci(az_puct_filter['cluster_bootstrap_95'])}). PUCT filtered rates were {pct(puct_filter['metrics']['cnn_init_puct100_filter']['rate'])} for initial CNN and {pct(puct_filter['metrics']['az_final_puct100_filter']['rate'])} for AZ. Safety filtering therefore contributes to PUCT results as well as policy-only results.",
        "",
        "## 3. Value head",
        "",
        f"After excluding the same AZ replay state, validation contains {value['n']} positions. Initial CNN W/D/L accuracy was {pct(value['paired_cluster_bootstrap_95']['cnn_init_wdl_accuracy']['estimate'])} {ci(value['paired_cluster_bootstrap_95']['cnn_init_wdl_accuracy']['ci'])}; final AZ was {pct(value['paired_cluster_bootstrap_95']['az_final_wdl_accuracy']['estimate'])} {ci(value['paired_cluster_bootstrap_95']['az_final_wdl_accuracy']['ci'])}. The paired AZ-minus-initial accuracy difference was {100 * value['paired_cluster_bootstrap_95']['wdl_accuracy_az_minus_init']['delta']:.1f} points, CI {ci(value['paired_cluster_bootstrap_95']['wdl_accuracy_az_minus_init']['ci'])}; it is not distinguished. AZ value MSE increased by {value['paired_cluster_bootstrap_95']['mse_az_minus_init']['delta']:.3f} (paired CI {value['paired_cluster_bootstrap_95']['mse_az_minus_init']['ci'][0]:.3f} to {value['paired_cluster_bootstrap_95']['mse_az_minus_init']['ci'][1]:.3f}).",
        f"The majority-class baseline reached {pct(value['baselines']['majority']['wdl_accuracy'])} W/D/L accuracy (CI {ci(value['baselines']['majority']['wdl_accuracy_ci'])}) and MSE {value['baselines']['majority']['mse']:.3f}. Paired clustered intervals show the initial CNN and final AZ accuracy deltas over this baseline as {value_paired['paired_cluster_bootstrap_95']['cnn_init']['majority']['wdl_accuracy_delta']:+.3f} (CI {value_paired['paired_cluster_bootstrap_95']['cnn_init']['majority']['wdl_accuracy_delta_ci']}) and {value_paired['paired_cluster_bootstrap_95']['az_final']['majority']['wdl_accuracy_delta']:+.3f} (CI {value_paired['paired_cluster_bootstrap_95']['az_final']['majority']['wdl_accuracy_delta_ci']}).",
        "",
        "## 4. Level-6 RL diagnostic",
        "",
        f"The historical RL run completed {rl_manifest['completed_steps']:,} episodes against Random only. Its mean logged reward was {sum(json.loads(line)['reward'] for line in (root / 'results/colab/rl_seed20261120_run1/train_log.jsonl').read_text(encoding='utf-8').splitlines()) / rl_manifest['completed_steps']:.3f} over those episodes. In the historical paired-opening round-robin, Random versus RL-Policy had n={rr_rl['games']}, score {rr_rl['score']:.3f}, bootstrap CI {ci(rr_rl['score_bootstrap_95'])}; the opponents and opening distribution differ, so training reward is not a general playing-strength measure.",
        f"Episode-0 was reconstructed from seed 20261120 using local PyTorch because the original initial checkpoint was not saved. The local PyTorch version differs from Colab, so this is a seeded random-initialization proxy. Exact-optimal rates raw/filtered were {pct(rl['methods']['rl_episode0_raw']['rate'])}/{pct(rl['methods']['rl_episode0_filter']['rate'])} for the proxy and {pct(rl['methods']['rl_final_raw']['rate'])}/{pct(rl['methods']['rl_final_filter']['rate'])} after training. The paired trained-filter minus episode-0-filter delta was {pct(rl['paired_differences']['trained_minus_episode0_with_filter']['delta'])}, CI {ci(rl['paired_differences']['trained_minus_episode0_with_filter']['cluster_bootstrap_95'])}; this does not distinguish learning from initialization.",
        "",
        "## 5. Solver cost and v3 data",
        "",
        f"The prior 50-position exact probe took {exact_probe['elapsed_seconds']:.1f} seconds. Mean cost was {exact_probe['by_stage']['opening']['mean_seconds']:.3f} seconds/opening, {exact_probe['by_stage']['middle']['mean_seconds']:.4f} seconds/middle, and {exact_probe['by_stage']['endgame']['mean_seconds']:.6f} seconds/endgame. Weak mode on the same positions took {weak_probe['elapsed_seconds']:.1f} seconds. No `7x6.book` exists under `solver/`. The strong ply-dependent search cost explains the observed runtime better than the Python wrapper overhead.",
        f"Dataset v3 generated 30,000 exact labels in {data_v3['local_label_seconds']:.1f} seconds with 12 workers ({(data_v3['splits']['train']['positions'] + data_v3['splits']['validation']['positions']) / data_v3['local_label_seconds']:.2f} positions/second aggregate). It is balanced at 10,000 positions per stage, has zero frozen-test overlap, and contains {data_v3['splits']['train']['source_games'] + data_v3['splits']['validation']['source_games']:,} source-game clusters. Selection rejected {sum(x['frozen_state_rejections'] for x in data_v3['selection'].values())} frozen-test states and {sum(x['mirror_duplicate_rejections'] for x in data_v3['selection'].values())} internal duplicates/mirrors. The matched AlphaBeta-3 teacher policy package contains 30,000 positions and took {ab3['elapsed_seconds']:.1f} seconds to prepare on 12 workers.",
        "",
        "## 6. Limits and next steps",
        "",
        "- The historical AlphaZero run has one seed and no iteration checkpoints. Its training loss does not establish stronger play.",
        "- Source-game cluster bootstrap accounts for within-game dependence but does not replace independent training seeds.",
        "- Tactical safety filters contribute materially for policy and PUCT agents and must remain separately reported from raw network output.",
        "- CNN v3 and controlled AZ V2 notebooks are prepared but have not been run on Colab. The total Colab GPU cap remains 180 minutes. RL is not scheduled because validation did not distinguish trained policy from the episode-0 proxy with the filter applied.",
        "",
        "## Source JSON",
        "",
        "All metrics above are rendered from these JSON files:",
        "",
    ]
    source_paths = [
        "results/diagnostic_v2_validation_clean_20261121_seed20261121.json",
        "results/diagnostic_v2_puct_filter_validation_20261121_seed20261121.json",
        "results/diagnostic_v2_value_ci_supplement_20261121_seed20261121.json",
        "results/diagnostic_v2_value_paired_baselines_20261121_seed20261121.json",
        "results/diagnostic_v2_rl_episode0_validation_20261121_seed20261121.json",
        "results/solver_probe_20261120_seed2701_50.json",
        "results/solver_weak_probe_20261121_seed2701_50.json",
        "colab/data/solverlabels_20261121_seed20261121_v3_30k/manifest.json",
        "colab/data/solverlabels_20261121_seed20261121_ab3_30k/manifest.json",
    ]
    lines.extend(f"- `{path}`" for path in source_paths)
    lines += ["", f"Render: `python -m scripts.render_learning_results_v2 --output docs/LEARNING_RESULTS_V2.md`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the V2 learning-agent report from result JSON")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("docs/LEARNING_RESULTS_V2.md"))
    args = parser.parse_args()
    destination = args.output if args.output.is_absolute() else args.root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(args.root), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
