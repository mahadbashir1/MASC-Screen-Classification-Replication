"""
Generate publication-quality comparison charts for MASC Paper Replication.
Produces: accuracy comparison, multi-metric comparison, and performance overview charts.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Agg')

# ─── Data from Paper (Table 5) and Replicated Results ──────────────────────────

classifiers = [
    'Gradient\nBoosting', 'SVM\nLinear', 'Logistic\nRegression*',
    'MLP*', 'Random\nForest', 'XGBoost',
    'Naive\nBayes*', 'Decision\nTree*', 'Adaboost', 'SVM\nRBF'
]

classifiers_short = [
    'GB', 'SVM-L', 'LR*', 'MLP*', 'RF', 'XGB',
    'NB*', 'DT*', 'Ada', 'SVM-R'
]

paper_accuracy   = [93.48, 93.20, 92.63, 93.20, 93.06, 93.20, 90.65, 92.35, 83.29, 81.16]
repl_accuracy    = [94.19, 94.05, 93.91, 93.91, 93.77, 93.63, 91.22, 90.93, 85.69, 83.71]

paper_f1    = [94.39, 94.27, 93.69, 94.13, 94.04, 93.99, 91.90, 93.01, 83.93, 80.90]
paper_roc   = [99.60, 99.48, 99.48, 99.56, 99.06, 99.51, 99.43, 96.91, 98.36, 98.04]

# Colors
PAPER_COLOR = '#4A90D9'
REPL_COLOR  = '#2ECC71'
ACCENT      = '#E74C3C'
BG_COLOR    = '#0D1117'
CARD_COLOR  = '#161B22'
TEXT_COLOR  = '#C9D1D9'
GRID_COLOR  = '#21262D'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': CARD_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'font.family': 'sans-serif',
    'font.size': 11,
})

os.makedirs('figures', exist_ok=True)

# ─── Chart 1: Grouped Bar — Paper vs Replicated Accuracy ──────────────────────

fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(classifiers))
width = 0.35

bars1 = ax.bar(x - width/2, paper_accuracy, width, label='Original Paper',
               color=PAPER_COLOR, edgecolor='white', linewidth=0.5, alpha=0.9)
bars2 = ax.bar(x + width/2, repl_accuracy, width, label='Replicated',
               color=REPL_COLOR, edgecolor='white', linewidth=0.5, alpha=0.9)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=PAPER_COLOR)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=REPL_COLOR)

ax.set_xlabel('Classifier', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Paper vs Replicated — Accuracy Comparison (All 10 Classifiers)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classifiers, fontsize=9)
ax.set_ylim(75, 100)
ax.legend(fontsize=12, loc='lower right', framealpha=0.8,
          facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/paper_vs_replicated_accuracy.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Generated: figures/paper_vs_replicated_accuracy.png")

# ─── Chart 2: Multi-Metric Radar/Comparison ───────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics_names = ['Accuracy', 'F1-Score', 'ROC-AUC']
paper_metrics = [paper_accuracy, paper_f1, paper_roc]
colors_metrics = ['#4A90D9', '#E67E22', '#9B59B6']

for idx, (ax, metric_name, paper_vals, color) in enumerate(
        zip(axes, metrics_names, paper_metrics, colors_metrics)):
    y_pos = np.arange(len(classifiers_short))
    bars = ax.barh(y_pos, paper_vals, height=0.6, color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.3)

    for i, (bar, val) in enumerate(zip(bars, paper_vals)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', ha='left', va='center', fontsize=8, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classifiers_short, fontsize=9)
    ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim(75, 105)
    ax.grid(axis='x', alpha=0.2)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Original Paper — Full Metrics Overview (Table 5)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/paper_metrics_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Generated: figures/paper_metrics_overview.png")

# ─── Chart 3: Difference Chart (Replicated - Paper) ──────────────────────────

fig, ax = plt.subplots(figsize=(14, 7))
differences = [r - p for r, p in zip(repl_accuracy, paper_accuracy)]
colors_diff = [REPL_COLOR if d >= 0 else ACCENT for d in differences]

y_pos = np.arange(len(classifiers_short))
bars = ax.barh(y_pos, differences, color=colors_diff, height=0.6,
               edgecolor='white', linewidth=0.5, alpha=0.9)

for i, (bar, diff, paper, repl) in enumerate(
        zip(bars, differences, paper_accuracy, repl_accuracy)):
    sign = '+' if diff >= 0 else ''
    label = f'{sign}{diff:.2f}%  ({paper:.2f}% → {repl:.2f}%)'
    x_pos = bar.get_width() + 0.05 if diff >= 0 else bar.get_width() - 0.05
    ha = 'left' if diff >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
            ha=ha, va='center', fontsize=9, fontweight='bold',
            color=colors_diff[i])

ax.axvline(x=0, color=TEXT_COLOR, linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(classifiers_short, fontsize=11, fontweight='bold')
ax.set_xlabel('Accuracy Difference (Replicated − Paper) in %', fontsize=12, fontweight='bold')
ax.set_title('Replication Accuracy Difference — How Close Are We?',
             fontsize=15, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=REPL_COLOR, label='Exceeded Paper'),
                   Patch(facecolor=ACCENT, label='Below Paper')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
          framealpha=0.8, facecolor=CARD_COLOR, edgecolor=GRID_COLOR)

plt.tight_layout()
plt.savefig('figures/accuracy_difference_chart.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Generated: figures/accuracy_difference_chart.png")

# ─── Chart 4: Side-by-Side Accuracy — Horizontal Bars ────────────────────────

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(classifiers_short))
height = 0.35

bars_paper = ax.barh(y_pos - height/2, paper_accuracy, height, label='Original Paper',
                     color=PAPER_COLOR, alpha=0.9, edgecolor='white', linewidth=0.3)
bars_repl = ax.barh(y_pos + height/2, repl_accuracy, height, label='Replicated',
                    color=REPL_COLOR, alpha=0.9, edgecolor='white', linewidth=0.3)

for bar in bars_paper:
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.2f}%', ha='left', va='center', fontsize=8, color=PAPER_COLOR)
for bar in bars_repl:
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.2f}%', ha='left', va='center', fontsize=8, color=REPL_COLOR)

ax.set_yticks(y_pos)
ax.set_yticklabels(classifiers_short, fontsize=11, fontweight='bold')
ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Model Accuracy: Original Paper vs Replication',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(75, 100)
ax.invert_yaxis()
ax.legend(fontsize=12, loc='lower right', framealpha=0.8,
          facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
ax.grid(axis='x', alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/model_accuracy_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Generated: figures/model_accuracy_comparison.png")

print("\nAll charts generated successfully in figures/ directory!")
