"""
fig_results.py — 막대 그래프 + 박스플롯 시각화
저장: results/figures/result_XXXXXXXX_fig.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re, glob, os, sys

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'results', 'logs')
FIG_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLOR = {8: '#16A34A', 16: '#E94B3C', 32: '#2563EB'}
ATTACK_LABEL = {'fgsm': 'FGSM', 'pgd': 'PGD', 'lavan': 'LaVAN', 'patch_fool': 'PatchFool'}
ATTACK_KEYS  = ['fgsm', 'pgd', 'lavan', 'patch_fool']


def parse_summary(content):
    m = re.search(r'Attack\s+((?:P=\d+\s*)+)', content)
    if not m:
        return {}, []
    p_values = [int(x) for x in re.findall(r'P=(\d+)', m.group(1))]

    results = {}
    for line in content.splitlines():
        line = line.strip()
        # attack 이름 추출 (첫 단어)
        parts = line.split()
        if not parts:
            continue
        atk = parts[0].lower()
        if atk not in ATTACK_KEYS:
            continue
        vals = re.findall(r'([\d.]+)%', line)
        if len(vals) == len(p_values):
            results[atk] = {p: float(v) for p, v in zip(p_values, vals)}

    return results, p_values


def parse_batch_frs(content, attack, p):
    pat = (rf"Attack:\s*{re.escape(attack)}\s*\|\s*P={p}.*?"
           rf"(?=─{{10}}|={10}|\Z)")
    sec = re.search(pat, content, re.DOTALL | re.IGNORECASE)
    if not sec:
        return []
    return [float(r) for r in re.findall(r'FR:\s*([\d.]+)%', sec.group())]


def main(log_path=None):
    if log_path is None:
        files = sorted(glob.glob(os.path.join(LOG_DIR, 'result_*.txt')))
        if not files:
            print("No result_*.txt found"); return
        log_path = files[-1]
    print(f"Reading: {log_path}")

    with open(log_path, encoding='utf-8') as f:
        content = f.read()

    summary, p_values = parse_summary(content)
    if not summary:
        print("SUMMARY 파싱 실패"); return

    attacks = [a for a in ATTACK_KEYS if a in summary]
    n_p     = len(p_values)
    print(f"공격: {attacks}  |  P값: {p_values}")

    batch_data = {(a, p): parse_batch_frs(content, a, p)
                  for a in attacks for p in p_values}
    has_batch  = any(len(v) > 1 for v in batch_data.values())

    ncols = 2 if has_batch else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    fig.suptitle('ViT Adversarial Robustness — Fooling Rate by Attack & Patch Size',
                 fontsize=13, fontweight='bold', y=1.01)

    # 막대 그래프
    ax1 = axes[0]
    x   = np.arange(len(attacks))
    w   = 0.7 / n_p
    off = np.linspace(-(n_p - 1) / 2, (n_p - 1) / 2, n_p) * w

    for pi, p in enumerate(p_values):
        vals = [summary[a].get(p, 0) for a in attacks]
        bars = ax1.bar(x + off[pi], vals, width=w * 0.9,
                       color=COLOR.get(p, '#888'), alpha=0.85,
                       label=f'P={p}', edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.8,
                     f'{v:.1f}%', ha='center', va='bottom',
                     fontsize=8, fontweight='bold',
                     color=COLOR.get(p, '#333'))

    ax1.set_xticks(x)
    ax1.set_xticklabels([ATTACK_LABEL.get(a, a) for a in attacks], fontsize=11)
    ax1.set_ylabel('Fooling Rate (%)', fontsize=11)
    ax1.set_title('Overall Fooling Rate', fontsize=11)
    ax1.set_ylim(0, 118)
    ax1.axhline(100, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 박스플롯
    if has_batch:
        ax2 = axes[1]
        positions, box_data, box_colors = [], [], []
        xtick_pos, xtick_lbl = [], []
        gap, inner = 1.3, 0.38

        for ai, atk in enumerate(attacks):
            center = ai * gap * n_p
            for pi, p in enumerate(p_values):
                bd = batch_data[(atk, p)]
                if bd:
                    positions.append(center + pi * inner)
                    box_data.append(bd)
                    box_colors.append(COLOR.get(p, '#888'))
            xtick_pos.append(center + (n_p - 1) * inner / 2)
            xtick_lbl.append(ATTACK_LABEL.get(atk, atk))

        bp = ax2.boxplot(box_data, positions=positions,
                         widths=inner * 0.7, patch_artist=True,
                         medianprops=dict(color='black', linewidth=1.8),
                         whiskerprops=dict(linewidth=1.2),
                         capprops=dict(linewidth=1.2),
                         flierprops=dict(marker='o', markersize=3, alpha=0.5))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for pos, bd, color in zip(positions, box_data, box_colors):
            ax2.scatter(pos, np.mean(bd), marker='D', color=color,
                        s=40, zorder=5, edgecolors='white', linewidth=0.8)

        ax2.set_xticks(xtick_pos)
        ax2.set_xticklabels(xtick_lbl, fontsize=11)
        ax2.set_ylabel('Fooling Rate (%)', fontsize=11)
        ax2.set_title('Batch-level FR Distribution  (◆ = mean)', fontsize=11)
        ax2.set_ylim(0, 118)
        ax2.axhline(100, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
        ax2.grid(axis='y', alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        legend_patches = [mpatches.Patch(color=COLOR.get(p, '#888'),
                          alpha=0.8, label=f'P={p}') for p in p_values]
        ax2.legend(handles=legend_patches, fontsize=10, loc='lower right')

    plt.tight_layout()
    out = os.path.join(FIG_DIR,
          os.path.basename(log_path).replace('.txt', '_fig.png'))
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


if __name__ == '__main__':
    main()