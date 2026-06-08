"""
tbl_results.py — 결과 비교 표 이미지 생성
저장: results/figures/result_XXXXXXXX_tbl.png
"""

import matplotlib.pyplot as plt
import numpy as np
import re, glob, os, sys

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'results', 'logs')
FIG_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

ATTACK_LABEL = {'fgsm': 'FGSM', 'pgd': 'PGD', 'lavan': 'LaVAN', 'patch_fool': 'PatchFool'}
ATTACK_KEYS  = ['fgsm', 'pgd', 'lavan', 'patch_fool']
P_ORDER      = [8, 16, 32]


def parse_log(content):
    m = re.search(r'Attack\s+((?:P=\d+\s*)+)', content)
    if not m:
        return {}, {}, []
    p_values = [int(x) for x in re.findall(r'P=(\d+)', m.group(1))]

    fr_results = {}
    for line in content.splitlines():
        line = line.strip()
        parts = line.split()
        if not parts:
            continue
        atk = parts[0].lower()
        if atk not in ATTACK_KEYS:
            continue
        vals = re.findall(r'([\d.]+)%', line)
        if len(vals) == len(p_values):
            fr_results[atk] = {p: float(v) for p, v in zip(p_values, vals)}

    # Clean Accuracy — P별 섹션 안에서 첫 번째 값 추출
    clean_acc = {}
    for p in p_values:
        pat = rf"P={p}\s*\|[^\n]*\n.*?Clean Accuracy[^:]*:\s*([\d.]+)%"
        cm  = re.search(pat, content, re.DOTALL)
        if cm:
            clean_acc[p] = float(cm.group(1))

    return fr_results, clean_acc, p_values


def value_to_color(val, vmin=30, vmax=100):
    norm = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    return (1.0, 1.0 - norm * 0.75, 1.0 - norm * 0.75)


def make_table(log_path, out_path):
    with open(log_path, encoding='utf-8') as f:
        content = f.read()

    fr_results, clean_acc, p_values = parse_log(content)
    if not fr_results:
        print(f"  파싱 실패: {log_path}"); return

    p_cols   = [p for p in P_ORDER if p in p_values]
    atk_rows = [a for a in ATTACK_KEYS if a in fr_results]
    n_col    = len(p_cols)

    col_labels = [f'P={p}\n(tokens={(224//p)**2})' for p in p_cols]
    row_labels, cell_text, cell_colors = [], [], []

    # Clean Accuracy 행
    if clean_acc:
        row_labels.append('Clean Accuracy')
        cell_text.append([f"{clean_acc.get(p, float('nan')):.1f}%" for p in p_cols])
        cell_colors.append(['#EFF6FF'] * n_col)

    # 공격별 FR 행
    for atk in atk_rows:
        row_labels.append(ATTACK_LABEL.get(atk, atk))
        vals = [fr_results[atk].get(p, float('nan')) for p in p_cols]
        cell_text.append([f'{v:.2f}%' if not np.isnan(v) else 'N/A' for v in vals])
        cell_colors.append([
            value_to_color(v) if not np.isnan(v) else (0.9, 0.9, 0.9)
            for v in vals
        ])

    # Δ 행
    if len(p_cols) >= 2:
        for i in range(len(p_cols) - 1):
            P1, P2 = p_cols[i], p_cols[i + 1]
            row_labels.append(f'Δ (P={P1} − P={P2})')
            delta_row, delta_colors = [], []
            for atk in atk_rows:
                v1 = fr_results[atk].get(P1, float('nan'))
                v2 = fr_results[atk].get(P2, float('nan'))
                if not np.isnan(v1) and not np.isnan(v2):
                    d = v1 - v2
                    delta_row.append(f'{d:+.2f}%p')
                    delta_colors.append('#DCFCE7' if d < 0 else '#FEE2E2')
                else:
                    delta_row.append('N/A')
                    delta_colors.append('#F3F4F6')
            cell_text.append(delta_row + [''] * (n_col - len(delta_row)))
            cell_colors.append(delta_colors + ['#F9FAFB'] * (n_col - len(delta_colors)))

    fig_h = 0.7 + 0.55 * len(row_labels)
    fig_w = 2.5 + 2.0 * n_col
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    timestamp = os.path.basename(log_path).replace('result_', '').replace('.txt', '')
    fig.suptitle(f'ViT Adversarial Robustness — Fooling Rate Summary\n'
                 f'(Tiny-ImageNet val, PatchFool CE_loss, iters=50)  [{timestamp}]',
                 fontsize=11, fontweight='bold', y=1.02)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    for j in range(n_col):
        tbl[(0, j)].set_facecolor('#1E3A5F')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(len(row_labels)):
        tbl[(i + 1, -1)].set_facecolor('#374151')
        tbl[(i + 1, -1)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'result_*.txt')))
    if not files:
        print("No result_*.txt found"); return

    print(f"총 {len(files)}개 로그 파일 처리 중...")
    for log_path in files:
        out = os.path.join(FIG_DIR,
              os.path.basename(log_path).replace('.txt', '_tbl.png'))
        print(f"  {os.path.basename(log_path)}")
        make_table(log_path, out)

    print("\n완료.")


if __name__ == '__main__':
    main()