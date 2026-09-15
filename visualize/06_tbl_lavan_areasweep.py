"""
06_tbl_lavan_areasweep.py — 실험 8 결과 표: LaVAN 공격 면적 스윕
06_fig_lavan_areasweep.py와 같은 데이터를 표로 보여준다.

사용법:
  python visualize/06_tbl_lavan_areasweep.py
"""
import os
from results_io import TABLE_DIR, gather_sweep_data
from tables import make_sweep_table
from conditions import LAVAN_AREASWEEP, LAVAN_AREASWEEP_P, LAVAN_AREASWEEP_TITLE


def main():
    x_values = [x for x, _ in LAVAN_AREASWEEP]
    x_labels = [f'{x}%' for x in x_values]
    data_by_p = gather_sweep_data('lavan', LAVAN_AREASWEEP_P, LAVAN_AREASWEEP)

    out = os.path.join(TABLE_DIR, '06_lavan_areasweep_table.png')
    make_sweep_table(x_values, x_labels, data_by_p, out, LAVAN_AREASWEEP_TITLE)


if __name__ == '__main__':
    main()