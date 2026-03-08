#!/usr/bin/env python3
"""
Steel Plates Fault Detection

Usage examples
--------------
# Full pipeline: EDA + training + all plots
python main.py

# EDA only (no model training)
python main.py --mode eda

# Train only (skip slow t-SNE / PCA plots)
python main.py --mode train

# Use a different config
python main.py --config my_config.yaml

# Override output directory
python main.py --output-dir results/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src import load_config, load, FaultClassifierTrainer, Plotter


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Steel Plates Fault Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--mode", choices=["all", "eda", "train"],
        default="all",
        help="all=eda+train | eda=plots only | train=skip EDA",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Override plots output directory",
    )
    p.add_argument(
        "--no-tsne", action="store_true",
        help="Skip t-SNE (slow on large datasets)",
    )
    return p.parse_args()


# PIPELINE STEPS
def run_eda(cfg, dataset, plotter, skip_tsne: bool = False) -> None:
    print("\n── EDA ──────────────────────────────────────────────────────")
    plotter.fig1_dataset_overview(dataset)
    plotter.fig2_feature_distributions(dataset, cfg.features.continuous_features)
    plotter.fig3_boxplots(dataset, cfg.features.boxplot_features)
    plotter.fig4_correlation(dataset)
    if not skip_tsne:
        plotter.fig5_dimensionality_reduction(dataset)
    else:
        print("  ⏭  fig5 skipped (--no-tsne)")


def run_training(cfg, dataset, plotter) -> None:
    print("\n── Model Training ───────────────────────────────────────────")
    trainer = FaultClassifierTrainer(cfg.models)
    results = trainer.fit_evaluate(dataset)

    print("\n  Computing permutation importance (RF)...")
    trainer.compute_permutation_importance(results["Random Forest"], dataset)

    best = FaultClassifierTrainer.best(results)
    print(f"\n  Best model : {best.name}  (F1={best.f1:.4f})")

    print("\n── Result Plots ─────────────────────────────────────────────")
    plotter.fig6_model_comparison(results, dataset)
    plotter.fig7_feature_importance(results, dataset)
    plotter.fig8_per_class_metrics(results, dataset)
    plotter.fig9_learning_curves(results, dataset)

    summary = {name: res.to_dict() for name, res in results.items()}
    out = Path(cfg.output.results_file)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out}")

    print("\n" + "=" * 60)
    print("  FINAL MODEL SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"\n  {name}")
        print(f"    Accuracy : {res.accuracy:.4f}")
        print(f"    F1       : {res.f1:.4f}")
        print(f"    ROC-AUC  : {res.roc_auc:.4f}")
        print(f"    CV Score : {res.cv_mean:.4f} ± {res.cv_std:.4f}")
    print(f"\n  Best Model : {best.name}  (F1 = {best.f1:.4f})")
    print("=" * 60)


# MAIN
def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    if args.output_dir:
        cfg.output.plot_dir = args.output_dir

    print(f"Mode    : {args.mode}")
    print(f"Data    : {cfg.data.path}")
    print(f"Plots → : {cfg.output.plot_dir}/")

    print("\n── Loading Data ─────────────────────────────────────────────")
    dataset = load(cfg.data)

    plotter = Plotter(cfg.style, cfg.output)

    if args.mode in ("all", "eda"):
        run_eda(cfg, dataset, plotter, skip_tsne=args.no_tsne)

    if args.mode in ("all", "train"):
        run_training(cfg, dataset, plotter)

    print(f"\nDone. All figures saved to: {cfg.output.plot_dir}/")


if __name__ == "__main__":
    main()
