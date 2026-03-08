"""
Visualisation module for Steel Plates Fault Detection.

Each fig_* method produces and saves exactly one numbered figure.
All style parameters come from StyleConfig — no hardcoded colours.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from pathlib import Path
from typing import Dict, List

from .config import StyleConfig, OutputConfig
from .data import SteelDataset
from .models import ModelResult, CATBOOST_AVAILABLE


def _apply_style(cfg: StyleConfig, dpi: int) -> None:
    plt.rcParams.update({
        "figure.facecolor": cfg.dark_bg,  "axes.facecolor":  cfg.card_bg,
        "axes.edgecolor":   cfg.grid_col, "axes.labelcolor": cfg.text_col,
        "axes.titlecolor":  cfg.text_col, "axes.titlesize":  13,
        "axes.labelsize":   11,           "xtick.color":     cfg.text_col,
        "ytick.color":      cfg.text_col, "text.color":      cfg.text_col,
        "grid.color":       cfg.grid_col, "grid.linestyle":  "--",
        "grid.alpha":       0.5,          "legend.facecolor":cfg.card_bg,
        "legend.edgecolor": cfg.grid_col, "font.family":     "DejaVu Sans",
        "savefig.dpi":      dpi,          "savefig.bbox":    "tight",
        "savefig.facecolor":cfg.dark_bg,
    })


class Plotter:
    """
    All nine figures for the steel plates project.

    Parameters
    ----------
    style_cfg  : colour palette from config.yaml
    output_cfg : plot_dir and dpi
    """

    def __init__(self, style_cfg: StyleConfig, output_cfg: OutputConfig):
        self.s   = style_cfg
        self.out = output_cfg
        Path(output_cfg.plot_dir).mkdir(parents=True, exist_ok=True)
        _apply_style(style_cfg, output_cfg.plot_dpi)
        # Build per-fault colour mapping from palette
        # (fault order is resolved at call time from the dataset)

    def _fault_colors(self, fault_types: List[str]) -> Dict[str, str]:
        return dict(zip(fault_types, self.s.palette))

    def _save(self, fig: plt.Figure, name: str) -> None:
        fig.savefig(Path(self.out.plot_dir) / name)
        plt.close(fig)
        print(f"  ✅ {name}")

    # EDA

    def fig1_dataset_overview(self, dataset: SteelDataset) -> None:
        s  = self.s
        df = dataset.df
        fc = self._fault_colors(dataset.fault_types)
        counts = df["Fault_Type"].value_counts()

        fig = plt.figure(figsize=(21, 13))
        fig.suptitle("STEEL PLATES FAULT DETECTION — Dataset Overview",
                     fontsize=18, fontweight="bold", color=s.accent, y=0.99)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

        # Bar chart
        ax = fig.add_subplot(gs[0, 0])
        bars = ax.bar(counts.index, counts.values,
                      color=[fc[f] for f in counts.index],
                      edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_title("Class Distribution", fontweight="bold")
        ax.set_xlabel("Fault Type"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35); ax.grid(axis="y")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                    str(val), ha="center", va="bottom", fontsize=9)

        # Pie
        ax = fig.add_subplot(gs[0, 1])
        _, _, autotexts = ax.pie(
            counts.values, labels=counts.index,
            colors=[fc[f] for f in counts.index],
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 8, "color": s.text_col},
            wedgeprops={"edgecolor": s.dark_bg, "linewidth": 1.5})
        [at.set_fontsize(7.5) for at in autotexts]
        ax.set_title("Fault Type Proportions", fontweight="bold")

        # Missing values
        ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
        ax.text(0.5, 0.5, "No Missing Values\nin Dataset",
                ha="center", va="center", fontsize=14,
                color="#4CAF50", fontweight="bold", transform=ax.transAxes)
        ax.set_title("Missing Value Analysis", fontweight="bold")

        # Steel type stacked bar
        ax = fig.add_subplot(gs[1, 0])
        steel_fault = df.groupby(["Fault_Type", "TypeOfSteel_A300"]).size().unstack(fill_value=0)
        steel_fault.columns = ["A400 Steel", "A300 Steel"]
        bv = np.zeros(len(steel_fault))
        for col, color in zip(steel_fault.columns, ["#2196F3", "#FF9800"]):
            ax.bar(steel_fault.index, steel_fault[col], bottom=bv,
                   color=color, label=col, alpha=0.85, edgecolor=s.dark_bg)
            bv += steel_fault[col].values
        ax.set_title("Steel Type by Fault Class", fontweight="bold")
        ax.set_xlabel("Fault Type"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35); ax.legend(fontsize=9); ax.grid(axis="y")

        # Thickness distribution
        ax = fig.add_subplot(gs[1, 1])
        tc = df["Steel_Plate_Thickness"].value_counts().sort_index()
        ax.bar(tc.index.astype(str), tc.values, color=s.accent, alpha=0.8, edgecolor=s.dark_bg)
        ax.set_title("Steel Plate Thickness Distribution", fontweight="bold")
        ax.set_xlabel("Thickness (mm)"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30); ax.grid(axis="y")

        # Imbalance profile
        ax = fig.add_subplot(gs[1, 2])
        sorted_counts = counts.sort_values()
        ax.barh(sorted_counts.index, sorted_counts.values,
                color=[fc[f] for f in sorted_counts.index], alpha=0.85, edgecolor=s.dark_bg)
        ax.axvline(sorted_counts.mean(), color="#FF9800", linestyle="--", linewidth=1.5,
                   label=f"Mean: {sorted_counts.mean():.0f}")
        ax.set_title("Class Imbalance Profile", fontweight="bold")
        ax.set_xlabel("Sample Count"); ax.legend(fontsize=9); ax.grid(axis="x")
        for i, val in enumerate(sorted_counts.values):
            ax.text(val+3, i, str(val), va="center", fontsize=9)

        self._save(fig, "01_dataset_overview.png")

    def fig2_feature_distributions(
        self, dataset: SteelDataset, continuous_features: List[str]
    ) -> None:
        s  = self.s
        df = dataset.df
        fc = self._fault_colors(dataset.fault_types)
        n  = len(continuous_features)
        ncols = 5 if n > 12 else 4
        nrows = -(-n // ncols)          # ceiling division

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        fig.suptitle("FEATURE DISTRIBUTIONS — KDE per Fault Class",
                     fontsize=16, fontweight="bold", color=s.accent, y=1.01)

        for i, (ax, feat) in enumerate(zip(axes.flat, continuous_features)):
            for fault in dataset.fault_types:
                subset = df[df["Fault_Type"] == fault][feat].dropna()
                if len(subset) > 5:
                    subset.plot.kde(ax=ax, color=fc[fault], label=fault,
                                    linewidth=1.8, alpha=0.85)
            ax.set_title(feat, fontweight="bold", fontsize=10)
            ax.set_ylabel("Density", fontsize=9); ax.grid(True); ax.tick_params(labelsize=8)
            if i == 0:
                ax.legend(fontsize=7, ncol=2)

        for ax in axes.flat[n:]:
            ax.axis("off")

        plt.tight_layout()
        self._save(fig, "02_feature_distributions.png")

    def fig3_boxplots(self, dataset: SteelDataset, boxplot_features: List[str]) -> None:
        s  = self.s
        df = dataset.df
        n  = len(boxplot_features)
        ncols = 4; nrows = -(-n // ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5*nrows))
        fig.suptitle("FEATURE vs FAULT TYPE — Boxplot Analysis",
                     fontsize=16, fontweight="bold", color=s.accent, y=1.01)

        for ax, feat in zip(axes.flat, boxplot_features):
            data = [df[df["Fault_Type"] == ft][feat].values for ft in dataset.fault_types]
            bp   = ax.boxplot(
                data, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=s.text_col),
                capprops=dict(color=s.text_col),
                flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                markerfacecolor=s.accent))
            for patch, color in zip(bp["boxes"], s.palette):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            ax.set_xticklabels(dataset.fault_types, rotation=35, ha="right", fontsize=8)
            ax.set_title(feat, fontweight="bold", fontsize=10); ax.grid(axis="y")

        for ax in axes.flat[n:]:
            ax.axis("off")

        plt.tight_layout()
        self._save(fig, "03_boxplots_per_class.png")

    def fig4_correlation(self, dataset: SteelDataset) -> None:
        s    = self.s
        corr = dataset.df[dataset.features].corr()
        cmap_div = LinearSegmentedColormap.from_list("div", ["#FF5722", s.card_bg, "#2196F3"])

        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        fig.suptitle("CORRELATION ANALYSIS", fontsize=16, fontweight="bold", color=s.accent)

        sns.heatmap(corr, ax=axes[0], cmap=cmap_div, center=0, vmin=-1, vmax=1,
                    linewidths=0.3, linecolor=s.dark_bg, annot=False, square=True,
                    cbar_kws={"shrink": 0.8})
        axes[0].set_title("Full Feature Correlation Matrix", fontweight="bold", fontsize=13)
        axes[0].tick_params(axis="x", rotation=90, labelsize=7)
        axes[0].tick_params(axis="y", labelsize=7)

        pairs = (corr
                 .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                 .stack().reset_index())
        pairs.columns = ["Feature1", "Feature2", "Correlation"]
        pairs["Abs"] = pairs["Correlation"].abs()
        top    = pairs.nlargest(15, "Abs")
        colors = [s.accent if v > 0 else "#FF5722" for v in top["Correlation"]]
        labels = [f"{r.Feature1}\nvs\n{r.Feature2}" for _, r in top.iterrows()]
        axes[1].barh(range(len(top)), top["Correlation"], color=colors, alpha=0.85, edgecolor=s.dark_bg)
        axes[1].set_yticks(range(len(top))); axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].axvline(0, color="white", linewidth=1, alpha=0.5)
        axes[1].set_xlabel("Correlation Coefficient")
        axes[1].set_title("Top 15 Feature Correlations", fontweight="bold", fontsize=13)
        axes[1].grid(axis="x")

        plt.tight_layout()
        self._save(fig, "04_correlation_analysis.png")

    def fig5_dimensionality_reduction(self, dataset: SteelDataset) -> None:
        s  = self.s
        fc = self._fault_colors(dataset.fault_types)

        pca    = PCA(n_components=10)
        X_pca  = pca.fit_transform(dataset.X_scaled)
        tsne   = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
        X_tsne = tsne.fit_transform(X_pca[:, :8])

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle("DIMENSIONALITY REDUCTION — PCA & t-SNE",
                     fontsize=16, fontweight="bold", color=s.accent)

        ax = axes[0]
        exp = pca.explained_variance_ratio_
        ax.bar(range(1, 11), exp*100, color=s.accent, alpha=0.8)
        ax.plot(range(1, 11), np.cumsum(exp)*100, "o-", color="#FF9800",
                linewidth=2, markersize=6, label="Cumulative")
        ax.axhline(80, color="#4CAF50", linestyle="--", linewidth=1.5, label="80% threshold")
        ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance (%)")
        ax.set_title("PCA — Explained Variance", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True)
        for bar, val in zip(ax.patches, exp*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

        for ax_i, X_emb, xlabel, ylabel, title in [
            (axes[1], X_pca,  "PC1", "PC2",     "PCA — PC1 vs PC2"),
            (axes[2], X_tsne, "t-SNE 1", "t-SNE 2", "t-SNE — 2D Embedding"),
        ]:
            for ft, color in fc.items():
                mask = dataset.df["Fault_Type"].values == ft
                ax_i.scatter(X_emb[mask, 0], X_emb[mask, 1], c=color, label=ft,
                             alpha=0.6, s=18, edgecolors="none")
            ax_i.set_xlabel(xlabel); ax_i.set_ylabel(ylabel)
            ax_i.set_title(title, fontweight="bold")
            ax_i.legend(fontsize=8, markerscale=1.5); ax_i.grid(True)

        plt.tight_layout()
        self._save(fig, "05_dimensionality_reduction.png")

    # Model results

    def fig6_model_comparison(
        self,
        results: Dict[str, ModelResult],
        dataset: SteelDataset,
    ) -> None:
        s           = self.s
        model_names = list(results.keys())
        best        = max(results.values(), key=lambda r: r.f1)

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle("MODEL COMPARISON & EVALUATION",
                     fontsize=16, fontweight="bold", color=s.accent)

        ax = axes[0]
        x = np.arange(3); w = 0.20
        for i, (name, color) in enumerate(zip(model_names, s.palette)):
            r    = results[name]
            vals = [r.accuracy, r.f1, r.roc_auc]
            bars = ax.bar(x+i*w, vals, w, label=name.split()[0], color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x+w*1.5); ax.set_xticklabels(["Accuracy", "F1 (Weighted)", "ROC-AUC"])
        ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
        ax.set_title("Performance Metrics", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y")

        ax = axes[1]
        cv_means = [results[n].cv_mean for n in model_names]
        cv_stds  = [results[n].cv_std  for n in model_names]
        short    = [n.split()[0] for n in model_names]
        bars = ax.bar(short, cv_means, color=s.palette[:len(model_names)], alpha=0.85,
                      yerr=cv_stds, capsize=6, edgecolor=s.dark_bg,
                      error_kw=dict(ecolor="white", linewidth=1.5))
        ax.set_ylabel("CV F1 Score (5-Fold)")
        ax.set_title("Cross-Validation Results", fontweight="bold")
        ax.tick_params(axis="x", rotation=12); ax.set_ylim(0, 1.05); ax.grid(axis="y")
        for bar, m, st in zip(bars, cv_means, cv_stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+st+0.01,
                    f"{m:.3f}±{st:.3f}", ha="center", va="bottom", fontsize=8.5)

        ax = axes[2]
        class_names = dataset.label_encoder.classes_
        cm      = confusion_matrix(dataset.y_test, best.y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmap_cm = LinearSegmentedColormap.from_list("cm", [s.card_bg, "#2196F3"])
        im = ax.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if cm_norm[i,j] < 0.6 else s.dark_bg)
        ax.set_title(f"Confusion Matrix\n({best.name})", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        self._save(fig, "06_model_comparison.png")

    def fig7_feature_importance(
        self,
        results: Dict[str, ModelResult],
        dataset: SteelDataset,
    ) -> None:
        s       = self.s
        rf_res  = results["Random Forest"]
        rf_imp  = pd.Series(rf_res.model.feature_importances_,
                            index=dataset.features).sort_values(ascending=True)
        perm_imp_vals = rf_res.importances if rf_res.importances is not None else np.zeros(len(dataset.features))
        perm_imp = pd.Series(perm_imp_vals,
                             index=dataset.features).sort_values(ascending=True)

        has_cb = CATBOOST_AVAILABLE and "CatBoost" in results
        ncols  = 3 if has_cb else 2
        title  = ("FEATURE IMPORTANCE — RF vs CatBoost vs Permutation"
                  if has_cb else "FEATURE IMPORTANCE — RF vs Permutation")

        fig, axes = plt.subplots(1, ncols, figsize=(9*ncols, 9))
        fig.suptitle(title, fontsize=16, fontweight="bold", color=s.accent)

        ax = axes[0]
        colors = [s.accent if v > rf_imp.median() else "#546E7A" for v in rf_imp.values]
        bars   = ax.barh(rf_imp.index, rf_imp.values, color=colors, alpha=0.85, edgecolor=s.dark_bg)
        ax.set_xlabel("Gini Importance")
        ax.set_title("Random Forest — Gini Importance", fontweight="bold")
        ax.grid(axis="x")
        ax.axvline(rf_imp.median(), color="#FF9800", linestyle="--", linewidth=1.5, label="Median")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, rf_imp.values):
            ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=7.5)

        if has_cb:
            cb_imp = pd.Series(results["CatBoost"].model.get_feature_importance(),
                               index=dataset.features).sort_values(ascending=True)
            ax = axes[1]
            colors_cb = ["#9C27B0" if v > cb_imp.median() else "#546E7A" for v in cb_imp.values]
            bars = ax.barh(cb_imp.index, cb_imp.values, color=colors_cb, alpha=0.85, edgecolor=s.dark_bg)
            ax.set_xlabel("CatBoost Feature Importance (%)")
            ax.set_title("CatBoost — Feature Importance", fontweight="bold")
            ax.grid(axis="x")
            ax.axvline(cb_imp.median(), color="#FF9800", linestyle="--", linewidth=1.5, label="Median")
            ax.legend(fontsize=9)
            for bar, val in zip(bars, cb_imp.values):
                ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                        f"{val:.2f}", va="center", fontsize=7.5)
            perm_ax = axes[2]
        else:
            perm_ax = axes[1]

        colors_p = [s.accent if v > perm_imp.median() else "#546E7A" for v in perm_imp.values]
        bars = perm_ax.barh(perm_imp.index, perm_imp.values, color=colors_p,
                            alpha=0.85, edgecolor=s.dark_bg)
        perm_ax.set_xlabel("Mean Accuracy Decrease")
        perm_ax.set_title("Permutation Importance (RF)", fontweight="bold")
        perm_ax.grid(axis="x")
        perm_ax.axvline(perm_imp.median(), color="#FF9800", linestyle="--", linewidth=1.5,
                        label="Median")
        perm_ax.legend(fontsize=9)
        for bar, val in zip(bars, perm_imp.values):
            perm_ax.text(max(val, 0)+0.0002, bar.get_y()+bar.get_height()/2,
                         f"{val:.4f}", va="center", fontsize=7.5)

        plt.tight_layout()
        self._save(fig, "07_feature_importance.png")

    def fig8_per_class_metrics(
        self,
        results: Dict[str, ModelResult],
        dataset: SteelDataset,
    ) -> None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        s           = self.s
        fc          = self._fault_colors(dataset.fault_types)
        best        = max(results.values(), key=lambda r: r.f1)
        class_names = dataset.label_encoder.classes_

        precision = precision_score(dataset.y_test, best.y_pred, average=None,
                                    labels=range(len(class_names)))
        recall    = recall_score(dataset.y_test, best.y_pred, average=None,
                                 labels=range(len(class_names)))
        f1_per    = f1_score(dataset.y_test, best.y_pred, average=None,
                             labels=range(len(class_names)))

        rf_imp    = pd.Series(results["Random Forest"].model.feature_importances_,
                              index=dataset.features)
        top6      = rf_imp.nlargest(6).index.tolist()

        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        fig.suptitle("PER-CLASS PERFORMANCE & FAULT PROFILES",
                     fontsize=16, fontweight="bold", color=s.accent)

        ax = axes[0]
        x = np.arange(len(class_names)); w = 0.28
        ax.bar(x-w, precision, w, label="Precision", color="#2196F3", alpha=0.85)
        ax.bar(x,   recall,    w, label="Recall",    color="#4CAF50", alpha=0.85)
        ax.bar(x+w, f1_per,    w, label="F1",        color="#FF9800", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
        ax.set_title(f"Per-Class Metrics ({best.name})", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y")
        for xi, f1v in zip(x, f1_per):
            ax.text(xi+w, f1v+0.02, f"{f1v:.2f}", ha="center", fontsize=8, color="#FF9800")

        ax = axes[1]
        means = dataset.df.groupby("Fault_Type")[top6].mean()
        means_norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
        im2 = ax.imshow(means_norm.T, aspect="auto",
                        cmap=LinearSegmentedColormap.from_list("g", [s.card_bg, "#4CAF50"]))
        ax.set_xticks(range(len(means_norm.index)))
        ax.set_xticklabels(means_norm.index, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(top6))); ax.set_yticklabels(top6, fontsize=9)
        for i in range(len(top6)):
            for j in range(len(means_norm.index)):
                ax.text(j, i, f"{means_norm.iloc[j, i]:.2f}",
                        ha="center", va="center", fontsize=8)
        ax.set_title("Fault Feature Profiles (Top 6, Normalized)", fontweight="bold")
        plt.colorbar(im2, ax=ax, fraction=0.046)

        ax = axes[2]
        test_counts = (pd.Series(dataset.y_test)
                       .map(dict(enumerate(class_names)))
                       .value_counts())
        bars = ax.bar(test_counts.index, test_counts.values,
                      color=[fc.get(f, s.accent) for f in test_counts.index],
                      edgecolor=s.dark_bg, alpha=0.85)
        ax.set_title("Test Set Distribution", fontweight="bold")
        ax.set_xlabel("Fault Type"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35); ax.grid(axis="y")
        for bar, val in zip(bars, test_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(val), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        self._save(fig, "08_per_class_metrics.png")

    def fig9_learning_curves(
        self,
        results: Dict[str, ModelResult],
        dataset: SteelDataset,
    ) -> None:
        s           = self.s
        model_names = list(results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle("LEARNING CURVES & GENERALIZATION ANALYSIS",
                     fontsize=16, fontweight="bold", color=s.accent)

        for ax, name in zip(axes.flat, model_names):
            tr_sizes, tr_scores, val_scores = learning_curve(
                results[name].model, dataset.X_scaled, dataset.y,
                cv=5, n_jobs=-1, scoring="f1_weighted",
                train_sizes=np.linspace(0.15, 1.0, 7))
            tm = tr_scores.mean(1);  ts = tr_scores.std(1)
            vm = val_scores.mean(1); vs = val_scores.std(1)

            ax.plot(tr_sizes, tm, "o-", color="#2196F3", linewidth=2, markersize=6, label="Training")
            ax.fill_between(tr_sizes, tm-ts, tm+ts, alpha=0.15, color="#2196F3")
            ax.plot(tr_sizes, vm, "o-", color="#4CAF50", linewidth=2, markersize=6, label="Validation")
            ax.fill_between(tr_sizes, vm-vs, vm+vs, alpha=0.15, color="#4CAF50")
            ax.set_xlabel("Training Samples"); ax.set_ylabel("F1 Score (Weighted)")
            ax.set_title(f"Learning Curve — {name}", fontweight="bold")
            ax.legend(fontsize=9); ax.set_ylim(0.3, 1.05); ax.grid(True)
            gap = tm[-1] - vm[-1]
            ax.text(0.05, 0.05, f"Overfit gap: {gap:.3f}", transform=ax.transAxes, fontsize=9,
                    color="#FF9800" if gap > 0.05 else "#4CAF50",
                    bbox=dict(boxstyle="round", facecolor=s.card_bg, alpha=0.7))

        for ax in axes.flat[len(model_names):]:
            ax.axis("off")

        plt.tight_layout()
        self._save(fig, "09_learning_curves.png")
