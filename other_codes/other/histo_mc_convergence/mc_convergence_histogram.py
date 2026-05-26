"""
Monte Carlo Convergence Analysis
==================================
這段 code 直接接在 ParallelVecScript.py 之後執行。
需要先有: true_accuracies (numpy array, shape: (4845,))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp

# ── 0. 先存 ground truth（在 ParallelVecScript.py 裡加這一行）──────────────
# np.save('accuracies_nn.npy', accuracies.numpy())
# 或者如果已經有 true_accuracies：
# true_accuracies = np.load('accuracies_nn.npy')

# ── 1. 實驗參數設定 ───────────────────────────────────────────────────────────
N_TOTAL   = len(true_accuracies)   # = 4845
N_REPEATS = 50                     # 每個 K 重複幾次（越多越穩定，50 已足夠）
RNG_SEED  = 42

# K 值：用百分比定義，對應你 Proposal 裡的 1%, 5%, 10%, 25%
# 同時加入更細的值來畫出平滑的收斂曲線
sampling_rates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
K_VALUES = [max(2, int(N_TOTAL * r)) for r in sampling_rates]
print(f"Ground Truth: {N_TOTAL} subsets")
print(f"K values: {K_VALUES}")
print(f"對應百分比: {[f'{r:.0%}' for r in sampling_rates]}")

# ── 2. Monte Carlo 收斂實驗 ───────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)

results = {}   # { k: { 'w1': [...], 'ks_stat': [...], 'ks_pval': [...] } }

print(f"\n開始 Monte Carlo 收斂分析 ({N_REPEATS} 次重複)...")
print(f"{'K':>6} | {'採樣率':>6} | {'W1 平均':>9} | {'W1 標準差':>9} | {'KS p>0.05':>10}")
print("-" * 55)

for k, rate in zip(K_VALUES, sampling_rates):
    w1_list, ks_stat_list, ks_pval_list = [], [], []

    for _ in range(N_REPEATS):
        # 從 ground truth 裡隨機抽 K 個（不放回）
        sampled = true_accuracies[rng.choice(N_TOTAL, size=k, replace=False)]

        # Wasserstein-1 Distance
        w1 = wasserstein_distance(true_accuracies, sampled)
        w1_list.append(w1)

        # KS Test
        ks_stat, ks_pval = ks_2samp(true_accuracies, sampled)
        ks_stat_list.append(ks_stat)
        ks_pval_list.append(ks_pval)

    results[k] = {
        'rate':    rate,
        'w1':      np.array(w1_list),
        'ks_stat': np.array(ks_stat_list),
        'ks_pval': np.array(ks_pval_list),
    }

    pct_pass = (np.array(ks_pval_list) > 0.05).mean() * 100
    print(f"{k:>6} | {rate:>6.1%} | "
          f"{np.mean(w1_list):>9.5f} | "
          f"{np.std(w1_list):>9.5f} | "
          f"{pct_pass:>9.1f}%")

# ── 3. 結果儲存 ───────────────────────────────────────────────────────────────
np.save('mc_convergence_results.npy', results)

# ── 4. Plot 1: W1 收斂曲線（主圖，論文用）────────────────────────────────────
k_arr    = np.array(K_VALUES)
rate_arr = np.array(sampling_rates) * 100   # 轉成百分比
w1_means = np.array([results[k]['w1'].mean() for k in K_VALUES])
w1_stds  = np.array([results[k]['w1'].std()  for k in K_VALUES])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左圖：W1 vs 採樣率
ax = axes[0]
ax.fill_between(rate_arr,
                w1_means - w1_stds,
                w1_means + w1_stds,
                alpha=0.2, color='steelblue', label='± 1 Std Dev')
ax.plot(rate_arr, w1_means, 'o-', color='steelblue',
        linewidth=2, markersize=6, label='W₁ (Mean)')

# 標記每個點的 K 值
for k, rate, w1m in zip(K_VALUES, rate_arr, w1_means):
    ax.annotate(f'K={k}', (rate, w1m),
                textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=8, color='steelblue')

ax.set_xlabel('Sampling Rate (%)', fontsize=12)
ax.set_ylabel('Wasserstein Distance W₁', fontsize=12)
ax.set_title(f'Monte Carlo Convergence: Histogram Baseline\n'
             f'(N={N_TOTAL} subsets, {N_REPEATS} repeats per K)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 右圖：KS Test p-value 通過率
ax2 = axes[1]
pval_pass = np.array(
    [(results[k]['ks_pval'] > 0.05).mean() * 100 for k in K_VALUES]
)
ax2.plot(rate_arr, pval_pass, 's-', color='seagreen',
         linewidth=2, markersize=6, label='% runs with p > 0.05')
ax2.axhline(95, color='gray', linestyle='--', alpha=0.6,
            label='95% threshold')
ax2.set_xlabel('Sampling Rate (%)', fontsize=12)
ax2.set_ylabel('Runs with p > 0.05 (%)', fontsize=12)
ax2.set_title('KS Test: When is Q statistically\nindistinguishable from P?', fontsize=12)
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('MC_Convergence_Histogram.png', dpi=150, bbox_inches='tight')
print("\nPlot 1 saved: MC_Convergence_Histogram.png")

# ── 5. Plot 2: Boxplot（顯示每個 K 的變異數）──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

box_data   = [results[k]['w1'] for k in K_VALUES]
box_labels = [f'{r:.0%}\n(K={k})' for k, r in zip(K_VALUES, sampling_rates)]

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                medianprops=dict(color='red', linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.5)

ax.set_xlabel('Sampling Rate (K subsets)', fontsize=12)
ax.set_ylabel('Wasserstein Distance W₁', fontsize=12)
ax.set_title(f'Distribution of W₁ over {N_REPEATS} Repeats\n'
             f'(Shows stability of Histogram Estimation)', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('MC_Convergence_Boxplot.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved: MC_Convergence_Boxplot.png")

# ── 6. Plot 3: 不同 K 的 Histogram 比較（直觀展示）──────────────────────────
selected_rates = [0.01, 0.05, 0.10, 0.25]
selected_k     = [max(2, int(N_TOTAL * r)) for r in selected_rates]

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
bins = np.linspace(0, 1, 30)
rng_plot = np.random.default_rng(99)

for ax, k, rate in zip(axes, selected_k, selected_rates):
    sample = true_accuracies[rng_plot.choice(N_TOTAL, size=k, replace=False)]
    w1_val = wasserstein_distance(true_accuracies, sample)

    ax.hist(true_accuracies, bins=bins, alpha=0.45, density=True,
            color='steelblue', label='P (Ground Truth)', edgecolor='black')
    ax.hist(sample, bins=bins, alpha=0.6, density=True,
            color='darkorange', label=f'Q (K={k})', edgecolor='darkorange')

    ax.set_title(f'Sampling Rate: {rate:.0%}\nK={k},  W₁={w1_val:.4f}', fontsize=11)
    ax.set_xlabel('Accuracy')
    if ax == axes[0]:
        ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.suptitle('Histogram Approximation Q vs. Ground Truth P at Different Sampling Rates',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('MC_Histogram_Comparison.png', dpi=150, bbox_inches='tight')
print("Plot 3 saved: MC_Histogram_Comparison.png")

# ── 7. Summary Table ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Rate':>6} | {'K':>5} | {'W1 Mean':>9} | {'W1 Std':>8} | "
      f"{'KS p>0.05':>10} | {'Min W1':>8} | {'Max W1':>8}")
print("-"*65)
for k, rate in zip(K_VALUES, sampling_rates):
    r = results[k]
    pct = (r['ks_pval'] > 0.05).mean() * 100
    print(f"{rate:>6.1%} | {k:>5} | {r['w1'].mean():>9.5f} | "
          f"{r['w1'].std():>8.5f} | {pct:>9.1f}% | "
          f"{r['w1'].min():>8.5f} | {r['w1'].max():>8.5f}")
print("="*65)
print(f"\n結論：W₁ 在採樣率約 ___% 時開始穩定（elbow point）")
print(f"      → 論文結論：用 Histogram Estimation，需要約 ___% 的子集才能可靠近似分佈")
