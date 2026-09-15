"""
conditions.py — ablation 비교 조건(CONDITIONS) 정의 모음
각 실험의 fig_*.py(그래프)와 tbl_*.py(표) 스크립트가 똑같은 CONDITIONS를 공유해야 해서
(둘 다 숫자로 시작하는 모듈이라 서로 import 불가) 여기 한곳에 모아둔다.

포맷: [(label, attack, P, tag, hatch, alpha), ...]
      hatch/alpha는 그래프 전용 스타일이고 표에서는 무시된다.
"""

# 실험 7: LaVAN 위치 고정
LAVAN_FIXEDLOC = [
    ('P=8\nRandom loc',  'lavan', 8,  '',        '...', 0.55),
    ('P=8\nFixed loc',   'lavan', 8,  'fixedloc', '...', 0.9),
    ('P=16\nRandom loc', 'lavan', 16, '',        '',    0.7),
    ('P=16\nFixed loc',  'lavan', 16, 'fixedloc', '',    1.0),
]
LAVAN_FIXEDLOC_TITLE = ('LaVAN: Random vs. Fixed (Center) Attack Location\n'
                        '(does the P=8 vs P=16 gap hold once position variance is removed?)')

# 실험 4+5: PatchFool 면적 통제 (scattered vs contiguous)
PATCHFOOL_AREAMATCH = [
    ('P=8\n1 token (64px)',        'patch_fool', 8,  '',                  '...', 0.5),
    ('P=8\n4 tokens, scattered',   'patch_fool', 8,  'areamatch16',       '',    0.7),
    ('P=8\n4 tokens, contiguous',  'patch_fool', 8,  'areamatch16contig', 'xxx', 0.85),
    ('P=16\n1 token (256px)',      'patch_fool', 16, '',                  '///', 1.0),
]
PATCHFOOL_AREAMATCH_TITLE = (
    'PatchFool: Patch-Size Robustness vs. Perturbed-Area / Compactness Confound\n'
    '(is P=8 still more robust once attacked pixel area — and its compactness — is matched?)')

# 실험 6: PatchFool 토큰 선택 방식 (Attn vs Rand)
PATCHFOOL_RANDSEL = [
    ('P=8\nAttn-select',  'patch_fool', 8,  '',        '...', 0.55),
    ('P=8\nRand-select',  'patch_fool', 8,  'randsel', '...', 0.9),
    ('P=16\nAttn-select', 'patch_fool', 16, '',        '',    0.7),
    ('P=16\nRand-select', 'patch_fool', 16, 'randsel', '',    1.0),
]
PATCHFOOL_RANDSEL_TITLE = ('PatchFool: Attention-Guided vs. Random Token Selection\n'
                          '(does smart targeting matter, or is attacking a token enough by itself?)')

# 실험 8: LaVAN 공격 면적 스윕 (x, tag) — x는 이미지 대비 패치 면적 %, 오름차순.
# 2%는 baseline(tag='')과 동일해서 재사용 — scripts/06_lavan_areasweep.sh 참고.
LAVAN_AREASWEEP = [
    (0.5, 'lavanarea5'),
    (1,   'lavanarea10'),
    (2,   ''),
    (5,   'lavanarea50'),
    (10,  'lavanarea100'),
]
LAVAN_AREASWEEP_P      = [8, 16, 32]
LAVAN_AREASWEEP_XLABEL = 'LaVAN patch area (% of image)'
LAVAN_AREASWEEP_TITLE  = ('LaVAN: Robustness vs. Attack Area\n'
                         '(does the P=8 vs P=16/32 gap flip at a different attack budget?)')