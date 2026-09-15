"""
06_fig_lavan_areasweep.py — 실험 8 시각화: LaVAN 공격 면적 스윕
scripts/06_lavan_areasweep.sh 결과를 시각화한다.
LaVAN은 patch_size를 모르고 항상 고정 물리 면적만 공격하는데, 그 면적을 0.5%~10%로
바꿔가며 P=8/16/32의 강건성 순위가 뒤바뀌는 지점(crossover)이 있는지 확인한다.

사용법:
  python visualize/06_fig_lavan_areasweep.py
"""
import os
from results_io import FIG_DIR, gather_sweep_data
from plotting import plot_sweep
from conditions import (LAVAN_AREASWEEP, LAVAN_AREASWEEP_P,
                        LAVAN_AREASWEEP_XLABEL, LAVAN_AREASWEEP_TITLE)


def main():
    x_values = [x for x, _ in LAVAN_AREASWEEP]
    data_by_p = gather_sweep_data('lavan', LAVAN_AREASWEEP_P, LAVAN_AREASWEEP)

    for p in LAVAN_AREASWEEP_P:
        print(f"P={p}: RA={data_by_p[p]['RA']}  ASR={data_by_p[p]['ASR']}")

    out = os.path.join(FIG_DIR, '06_lavan_areasweep.png')
    plot_sweep(x_values, data_by_p, out, LAVAN_AREASWEEP_TITLE,
              LAVAN_AREASWEEP_XLABEL, attack='lavan')


if __name__ == '__main__':
    main()