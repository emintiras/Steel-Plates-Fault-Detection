import numpy as np
import pandas as pds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (confusion_matrix, roc_auc_score, accuracy_score,
                              f1_score, precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
import os

import warnings
warnings.filterwarnings('ignore')


try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("WARNING: CatBoost not installed. Run: pip install catboost")
    print("CatBoost will be skipped; all other models will still run.\n")

# OUTPUT DIRECTORY
OUT = 'plots'
os.makedirs(OUT, exist_ok=True)

# STYLE
PALETTE  = ['#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800','#00BCD4','#E91E63']
DARK_BG  = '#0d1117'
CARD_BG  = '#161b22'
TEXT_COL = '#e6edf3'
GRID_COL = '#21262d'
ACCENT   = '#58a6ff'

plt.rcParams.update({
    'figure.facecolor' : DARK_BG,  'axes.facecolor'   : CARD_BG,
    'axes.edgecolor'   : GRID_COL, 'axes.labelcolor'  : TEXT_COL,
    'axes.titlecolor'  : TEXT_COL, 'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,       'xtick.color'      : TEXT_COL,
    'ytick.color'      : TEXT_COL, 'text.color'       : TEXT_COL,
    'grid.color'       : GRID_COL, 'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,      'legend.facecolor' : CARD_BG,
    'legend.edgecolor' : GRID_COL, 'font.family'      : 'DejaVu Sans',
    'savefig.dpi'      : 150,      'savefig.bbox'     : 'tight',
    'savefig.facecolor': DARK_BG,
})

FAULT_COLS   = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
FAULT_COLORS = dict(zip(FAULT_COLS, PALETTE))

# 1. LOAD & PREPARE DATA
df_raw = pd.read_csv('data/steel_plates_faults.csv', index_col=0)

# Feature columns = everything except the 7 fault label columns
FEATURES = [c for c in df_raw.columns if c not in FAULT_COLS]

# Convert one-hot labels to single Fault_Type column
df_raw['Fault_Type'] = df_raw[FAULT_COLS].idxmax(axis=1)
df = df_raw[FEATURES + ['Fault_Type']].copy()

FAULT_TYPES = FAULT_COLS   # preserve canonical order

print(f"Dataset: {df.shape[0]} rows x {len(FEATURES)} features x {len(FAULT_COLS)} classes")
print(df['Fault_Type'].value_counts())

# 2. FIGURE 1 — DATASET OVERVIEW
fig = plt.figure(figsize=(21, 13))
fig.suptitle('STEEL PLATES FAULT DETECTION — Dataset Overview',
             fontsize=18, fontweight='bold', color=ACCENT, y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

counts = df['Fault_Type'].value_counts()

ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(counts.index, counts.values,
               color=[FAULT_COLORS[f] for f in counts.index],
               edgecolor='white', linewidth=0.5, alpha=0.9)
ax1.set_title('Class Distribution', fontweight='bold')
ax1.set_xlabel('Fault Type'); ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=35); ax1.grid(axis='y')
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
             str(val), ha='center', va='bottom', fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(
    counts.values, labels=counts.index,
    colors=[FAULT_COLORS[f] for f in counts.index],
    autopct='%1.1f%%', startangle=140,
    textprops={'fontsize': 8, 'color': TEXT_COL},
    wedgeprops={'edgecolor': DARK_BG, 'linewidth': 1.5})
[at.set_fontsize(7.5) for at in autotexts]
ax2.set_title('Fault Type Proportions', fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.5, 'No Missing Values\nin Dataset',
         ha='center', va='center', fontsize=14,
         color='#4CAF50', fontweight='bold', transform=ax3.transAxes)
ax3.set_title('Missing Value Analysis', fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(gs[1, 0])
steel_fault = df.groupby(['Fault_Type','TypeOfSteel_A300']).size().unstack(fill_value=0)
steel_fault.columns = ['A400 Steel','A300 Steel']
bv = np.zeros(len(steel_fault))
for col, color in zip(steel_fault.columns, ['#2196F3','#FF9800']):
    ax4.bar(steel_fault.index, steel_fault[col], bottom=bv,
            color=color, label=col, alpha=0.85, edgecolor=DARK_BG)
    bv += steel_fault[col].values
ax4.set_title('Steel Type by Fault Class', fontweight='bold')
ax4.set_xlabel('Fault Type'); ax4.set_ylabel('Count')
ax4.tick_params(axis='x', rotation=35); ax4.legend(fontsize=9); ax4.grid(axis='y')

ax5 = fig.add_subplot(gs[1, 1])
tc = df['Steel_Plate_Thickness'].value_counts().sort_index()
ax5.bar(tc.index.astype(str), tc.values, color=ACCENT, alpha=0.8, edgecolor=DARK_BG)
ax5.set_title('Steel Plate Thickness Distribution', fontweight='bold')
ax5.set_xlabel('Thickness (mm)'); ax5.set_ylabel('Count')
ax5.tick_params(axis='x', rotation=30); ax5.grid(axis='y')

ax6 = fig.add_subplot(gs[1, 2])
sorted_counts = counts.sort_values()
ax6.barh(sorted_counts.index, sorted_counts.values,
         color=[FAULT_COLORS[f] for f in sorted_counts.index], alpha=0.85, edgecolor=DARK_BG)
ax6.axvline(sorted_counts.mean(), color='#FF9800', linestyle='--', linewidth=1.5,
            label=f'Mean: {sorted_counts.mean():.0f}')
ax6.set_title('Class Imbalance Profile', fontweight='bold')
ax6.set_xlabel('Sample Count'); ax6.legend(fontsize=9); ax6.grid(axis='x')
for i, val in enumerate(sorted_counts.values):
    ax6.text(val+3, i, str(val), va='center', fontsize=9)

fig.savefig(f'{OUT}/01_dataset_overview.png')
plt.close(fig)
print("Figure 1 saved")

# 3. FIGURE 2 — FEATURE DISTRIBUTIONS
continuous_features = ['Pixels_Areas','Sum_of_Luminosity','LogOfAreas','Luminosity_Index',
                       'SigmoidOfAreas','Edges_Index','Empty_Index','Square_Index',
                       'Outside_X_Index','Log_X_Index','Log_Y_Index','Orientation_Index']

fig, axes = plt.subplots(3, 4, figsize=(22, 13))
fig.suptitle('FEATURE DISTRIBUTIONS — KDE per Fault Class',
             fontsize=16, fontweight='bold', color=ACCENT, y=1.01)

for i, (ax, feat) in enumerate(zip(axes.flat, continuous_features)):
    for fault in FAULT_TYPES:
        subset = df[df['Fault_Type'] == fault][feat].dropna()
        if len(subset) > 5:
            subset.plot.kde(ax=ax, color=FAULT_COLORS[fault], label=fault,
                            linewidth=1.8, alpha=0.85)
    ax.set_title(feat, fontweight='bold', fontsize=10)
    ax.set_ylabel('Density', fontsize=9); ax.grid(True); ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
fig.savefig(f'{OUT}/02_feature_distributions.png')
plt.close(fig)
print("Figure 2 saved")

# 4. FIGURE 3 — BOXPLOTS PER CLASS
box_features = ['Pixels_Areas','LogOfAreas','Luminosity_Index','Edges_Index',
                'Empty_Index','Square_Index','Orientation_Index','SigmoidOfAreas']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('FEATURE vs FAULT TYPE — Boxplot Analysis',
             fontsize=16, fontweight='bold', color=ACCENT, y=1.01)

for ax, feat in zip(axes.flat, box_features):
    data = [df[df['Fault_Type'] == ft][feat].values for ft in FAULT_TYPES]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color=TEXT_COL), capprops=dict(color=TEXT_COL),
                    flierprops=dict(marker='o', markersize=2, alpha=0.3,
                                    markerfacecolor=ACCENT))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(FAULT_TYPES, rotation=35, ha='right', fontsize=8)
    ax.set_title(feat, fontweight='bold', fontsize=10); ax.grid(axis='y')

plt.tight_layout()
fig.savefig(f'{OUT}/03_boxplots_per_class.png')
plt.close(fig)
print("Figure 3 saved")

# 5. FIGURE 4 — CORRELATION ANALYSIS
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('CORRELATION ANALYSIS', fontsize=16, fontweight='bold', color=ACCENT)

corr = df[FEATURES].corr()
cmap_div = LinearSegmentedColormap.from_list('div', ['#FF5722', CARD_BG, '#2196F3'])
sns.heatmap(corr, ax=axes[0], cmap=cmap_div, center=0, vmin=-1, vmax=1,
            linewidths=0.3, linecolor=DARK_BG, annot=False, square=True,
            cbar_kws={'shrink': 0.8})
axes[0].set_title('Full Feature Correlation Matrix', fontweight='bold', fontsize=13)
axes[0].tick_params(axis='x', rotation=90, labelsize=7)
axes[0].tick_params(axis='y', labelsize=7)

pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
pairs.columns = ['Feature1','Feature2','Correlation']
pairs['Abs'] = pairs['Correlation'].abs()
top = pairs.nlargest(15, 'Abs')
colors_b = [ACCENT if v > 0 else '#FF5722' for v in top['Correlation']]
labels = [f"{r.Feature1}\nvs\n{r.Feature2}" for _, r in top.iterrows()]
axes[1].barh(range(len(top)), top['Correlation'], color=colors_b, alpha=0.85, edgecolor=DARK_BG)
axes[1].set_yticks(range(len(top))); axes[1].set_yticklabels(labels, fontsize=7)
axes[1].axvline(0, color='white', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Correlation Coefficient')
axes[1].set_title('Top 15 Feature Correlations', fontweight='bold', fontsize=13)
axes[1].grid(axis='x')

plt.tight_layout()
fig.savefig(f'{OUT}/04_correlation_analysis.png')
plt.close(fig)
print("Figure 4 saved")

# 6. FIGURE 5 — PCA + t-SNE
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(df[FEATURES])
le        = LabelEncoder()
y         = le.fit_transform(df['Fault_Type'])

pca   = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

tsne   = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
X_tsne = tsne.fit_transform(X_pca[:, :8])

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('DIMENSIONALITY REDUCTION — PCA & t-SNE',
             fontsize=16, fontweight='bold', color=ACCENT)

ax = axes[0]
exp = pca.explained_variance_ratio_; cum = np.cumsum(exp)
bars = ax.bar(range(1, 11), exp * 100, color=ACCENT, alpha=0.8)
ax.plot(range(1, 11), cum * 100, 'o-', color='#FF9800', linewidth=2, markersize=6, label='Cumulative')
ax.axhline(80, color='#4CAF50', linestyle='--', linewidth=1.5, label='80% threshold')
ax.set_xlabel('Principal Component'); ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA — Explained Variance', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True)
for bar, val in zip(bars, exp * 100):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

ax = axes[1]
for ft, color in FAULT_COLORS.items():
    mask = df['Fault_Type'].values == ft
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=ft,
               alpha=0.6, s=18, edgecolors='none')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('PCA — PC1 vs PC2', fontweight='bold')
ax.legend(fontsize=8, markerscale=1.5); ax.grid(True)

ax = axes[2]
for ft, color in FAULT_COLORS.items():
    mask = df['Fault_Type'].values == ft
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=ft,
               alpha=0.6, s=18, edgecolors='none')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE — 2D Embedding', fontweight='bold')
ax.legend(fontsize=8, markerscale=1.5); ax.grid(True)

plt.tight_layout()
fig.savefig(f'{OUT}/05_dimensionality_reduction.png')
plt.close(fig)
print("Figure 5 saved")

# 7. TRAIN MODELS
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Random Forest'      : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=20),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=150, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}
if CATBOOST_AVAILABLE:
    models['CatBoost'] = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        random_seed=42, verbose=0,
        loss_function='MultiClass', eval_metric='Accuracy'
    )

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    n_jobs_cv = 1 if name == 'CatBoost' else -1
    cv     = cross_val_score(model, X_scaled, y, cv=5, scoring='f1_weighted', n_jobs=n_jobs_cv)
    results[name] = {
        'model'   : model,
        'y_pred'  : y_pred,
        'y_prob'  : y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1'      : f1_score(y_test, y_pred, average='weighted'),
        'roc_auc' : roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'),
        'cv_mean' : cv.mean(),
        'cv_std'  : cv.std(),
    }
    print(f"  {name}: Acc={results[name]['accuracy']:.3f}  "
          f"F1={results[name]['f1']:.3f}  AUC={results[name]['roc_auc']:.3f}")

best_name   = max(results, key=lambda k: results[k]['f1'])
class_names = le.classes_
model_names = list(results.keys())

# 8. FIGURE 6 — MODEL COMPARISON
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('MODEL COMPARISON & EVALUATION',
             fontsize=16, fontweight='bold', color=ACCENT)

ax = axes[0]
x = np.arange(3); w = 0.20
for i, (name, color) in enumerate(zip(model_names, PALETTE[:4])):
    vals = [results[name]['accuracy'], results[name]['f1'], results[name]['roc_auc']]
    bars = ax.bar(x + i * w, vals, w, label=name.split()[0], color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x + w * 1.5); ax.set_xticklabels(['Accuracy','F1 (Weighted)','ROC-AUC'])
ax.set_ylim(0, 1.08); ax.set_ylabel('Score')
ax.set_title('Performance Metrics', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y')

ax = axes[1]
cv_means = [results[n]['cv_mean'] for n in model_names]
cv_stds  = [results[n]['cv_std']  for n in model_names]
short_names = [n.split()[0] for n in model_names]
bars = ax.bar(short_names, cv_means, color=PALETTE[:4], alpha=0.85,
              yerr=cv_stds, capsize=6, edgecolor=DARK_BG,
              error_kw=dict(ecolor='white', linewidth=1.5))
ax.set_ylabel('CV F1 Score (5-Fold)'); ax.set_title('Cross-Validation Results', fontweight='bold')
ax.tick_params(axis='x', rotation=12); ax.set_ylim(0, 1.05); ax.grid(axis='y')
for bar, m, s in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.01,
            f'{m:.3f}+/-{s:.3f}', ha='center', va='bottom', fontsize=8.5)

ax = axes[2]
cm      = confusion_matrix(y_test, results[best_name]['y_pred'])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cmap_cm = LinearSegmentedColormap.from_list('cm', [CARD_BG, '#2196F3'])
im = ax.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1)
ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                fontsize=8, color='white' if cm_norm[i,j] < 0.6 else DARK_BG)
ax.set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
fig.savefig(f'{OUT}/06_model_comparison.png')
plt.close(fig)
print("Figure 6 saved")

# 9. FIGURE 7 — FEATURE IMPORTANCE
rf_model    = results['Random Forest']['model']
feat_imp_rf = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
perm_imp    = permutation_importance(rf_model, X_test, y_test,
                                     n_repeats=10, random_state=42, n_jobs=-1)
perm_series = pd.Series(perm_imp.importances_mean, index=FEATURES).sort_values(ascending=True)

# Use RF importances as canonical for downstream (fault profiles, top6)
feat_imp = feat_imp_rf

if CATBOOST_AVAILABLE:
    cb_model    = results['CatBoost']['model']
    feat_imp_cb = pd.Series(cb_model.get_feature_importance(), index=FEATURES).sort_values(ascending=True)
    n_imp_cols  = 3
    fig_title   = 'FEATURE IMPORTANCE — RF vs CatBoost vs Permutation'
else:
    n_imp_cols  = 2
    fig_title   = 'FEATURE IMPORTANCE ANALYSIS — RF vs Permutation'

fig, axes = plt.subplots(1, n_imp_cols, figsize=(9 * n_imp_cols, 9))
fig.suptitle(fig_title, fontsize=16, fontweight='bold', color=ACCENT)

ax = axes[0]
colors_rf = [ACCENT if v > feat_imp_rf.median() else '#546E7A' for v in feat_imp_rf.values]
bars = ax.barh(feat_imp_rf.index, feat_imp_rf.values, color=colors_rf, alpha=0.85, edgecolor=DARK_BG)
ax.set_xlabel('Gini Importance')
ax.set_title('Random Forest — Gini Importance', fontweight='bold')
ax.grid(axis='x')
ax.axvline(feat_imp_rf.median(), color='#FF9800', linestyle='--', linewidth=1.5, label='Median')
ax.legend(fontsize=9)
for bar, val in zip(bars, feat_imp_rf.values):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7.5)

if CATBOOST_AVAILABLE:
    ax = axes[1]
    colors_cb = ['#9C27B0' if v > feat_imp_cb.median() else '#546E7A' for v in feat_imp_cb.values]
    bars = ax.barh(feat_imp_cb.index, feat_imp_cb.values, color=colors_cb, alpha=0.85, edgecolor=DARK_BG)
    ax.set_xlabel('CatBoost Feature Importance (%)')
    ax.set_title('CatBoost — Feature Importance', fontweight='bold')
    ax.grid(axis='x')
    ax.axvline(feat_imp_cb.median(), color='#FF9800', linestyle='--', linewidth=1.5, label='Median')
    ax.legend(fontsize=9)
    for bar, val in zip(bars, feat_imp_cb.values):
        ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=7.5)
    perm_ax = axes[2]
else:
    perm_ax = axes[1]

colors_perm = [ACCENT if v > perm_series.median() else '#546E7A' for v in perm_series.values]
bars = perm_ax.barh(perm_series.index, perm_series.values, color=colors_perm, alpha=0.85, edgecolor=DARK_BG)
perm_ax.set_xlabel('Mean Accuracy Decrease')
perm_ax.set_title('Permutation Importance (RF)', fontweight='bold')
perm_ax.grid(axis='x')
perm_ax.axvline(perm_series.median(), color='#FF9800', linestyle='--', linewidth=1.5, label='Median')
perm_ax.legend(fontsize=9)
for bar, val in zip(bars, perm_series.values):
    perm_ax.text(max(val, 0)+0.0002, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7.5)

plt.tight_layout()
fig.savefig(f'{OUT}/07_feature_importance.png')
plt.close(fig)
print("Figure 7 saved")

# 10. FIGURE 8 — PER-CLASS METRICS & FAULT PROFILES
y_pred_best = results[best_name]['y_pred']
precision   = precision_score(y_test, y_pred_best, average=None, labels=range(len(class_names)))
recall      = recall_score(y_test, y_pred_best, average=None, labels=range(len(class_names)))
f1_per      = f1_score(y_test, y_pred_best, average=None, labels=range(len(class_names)))

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle('PER-CLASS PERFORMANCE & FAULT PROFILES',
             fontsize=16, fontweight='bold', color=ACCENT)

ax = axes[0]
x = np.arange(len(class_names)); w = 0.28
ax.bar(x - w, precision, w, label='Precision', color='#2196F3', alpha=0.85)
ax.bar(x,     recall,    w, label='Recall',    color='#4CAF50', alpha=0.85)
ax.bar(x + w, f1_per,    w, label='F1',        color='#FF9800', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=8.5)
ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
ax.set_title(f'Per-Class Metrics ({best_name})', fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y')
for xi, f1v in zip(x, f1_per):
    ax.text(xi + w, f1v + 0.02, f'{f1v:.2f}', ha='center', fontsize=8, color='#FF9800')

top6 = feat_imp.tail(6).index.tolist()
ax   = axes[1]
means_per_class = df.groupby('Fault_Type')[top6].mean()
means_norm      = ((means_per_class - means_per_class.min()) /
                   (means_per_class.max() - means_per_class.min() + 1e-9))
im2 = ax.imshow(means_norm.T, aspect='auto',
                cmap=LinearSegmentedColormap.from_list('g', [CARD_BG, '#4CAF50']))
ax.set_xticks(range(len(means_norm.index)))
ax.set_xticklabels(means_norm.index, rotation=35, ha='right', fontsize=8.5)
ax.set_yticks(range(len(top6))); ax.set_yticklabels(top6, fontsize=9)
for i in range(len(top6)):
    for j in range(len(means_norm.index)):
        ax.text(j, i, f'{means_norm.iloc[j, i]:.2f}',
                ha='center', va='center', fontsize=8)
ax.set_title('Fault Feature Profiles (Top 6, Normalized)', fontweight='bold')
plt.colorbar(im2, ax=ax, fraction=0.046)

ax = axes[2]
test_counts = pd.Series(y_test).map(dict(enumerate(class_names))).value_counts()
bars = ax.bar(test_counts.index, test_counts.values,
              color=[FAULT_COLORS[f] for f in test_counts.index],
              edgecolor=DARK_BG, alpha=0.85)
ax.set_title('Test Set Distribution', fontweight='bold')
ax.set_xlabel('Fault Type'); ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=35); ax.grid(axis='y')
for bar, val in zip(bars, test_counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            str(val), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig.savefig(f'{OUT}/08_per_class_metrics.png')
plt.close(fig)
print("Figure 8 saved")

# 11. FIGURE 9 — LEARNING CURVES
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('LEARNING CURVES & GENERALIZATION ANALYSIS',
             fontsize=16, fontweight='bold', color=ACCENT)

for ax, name in zip(axes.flat, model_names):
    tr_sizes, tr_scores, val_scores = learning_curve(
        results[name]['model'], X_scaled, y,
        cv=5, n_jobs=-1, scoring='f1_weighted',
        train_sizes=np.linspace(0.15, 1.0, 7))
    tm = tr_scores.mean(1); ts = tr_scores.std(1)
    vm = val_scores.mean(1); vs = val_scores.std(1)

    ax.plot(tr_sizes, tm, 'o-', color='#2196F3', linewidth=2, markersize=6, label='Training')
    ax.fill_between(tr_sizes, tm - ts, tm + ts, alpha=0.15, color='#2196F3')
    ax.plot(tr_sizes, vm, 'o-', color='#4CAF50', linewidth=2, markersize=6, label='Validation')
    ax.fill_between(tr_sizes, vm - vs, vm + vs, alpha=0.15, color='#4CAF50')
    ax.set_xlabel('Training Samples'); ax.set_ylabel('F1 Score (Weighted)')
    ax.set_title(f'Learning Curve — {name}', fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0.3, 1.05); ax.grid(True)
    gap = tm[-1] - vm[-1]
    ax.text(0.05, 0.05, f'Overfit gap: {gap:.3f}', transform=ax.transAxes, fontsize=9,
            color='#FF9800' if gap > 0.05 else '#4CAF50',
            bbox=dict(boxstyle='round', facecolor=CARD_BG, alpha=0.7))

plt.tight_layout()
fig.savefig(f'{OUT}/09_learning_curves.png')
plt.close(fig)
print("Figure 9 saved")

# FINAL SUMMARY
print("\n" + "="*60)
print("  FINAL MODEL SUMMARY")
print("="*60)
for name, res in results.items():
    print(f"\n  {name}")
    print(f"    Accuracy : {res['accuracy']:.4f}")
    print(f"    F1       : {res['f1']:.4f}")
    print(f"    ROC-AUC  : {res['roc_auc']:.4f}")
    print(f"    CV Score : {res['cv_mean']:.4f} +/- {res['cv_std']:.4f}")

print(f"\n  Best Model : {best_name}  (F1 = {results[best_name]['f1']:.4f})")
print("="*60)
print(f"\nAll figures saved to: {OUT}/")