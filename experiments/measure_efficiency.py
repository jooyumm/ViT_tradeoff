"""
measure_efficiency.py
P=8/16/32 추론 속도, FLOPs, 파라미터 수 측정 + Fooling Rate 연동
→ 보안-효율 트레이드오프 표 출력 및 저장
"""

import torch
import time
import re
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model

PATCH_SIZES   = [8, 16, 32]
WARMUP_RUNS   = 10
MEASURE_RUNS  = 100
BATCH_SIZE    = 1
RESULTS_ROOT  = os.path.join(ROOT, 'results')
LOG_DIR       = os.path.join(RESULTS_ROOT, 'logs')
PROFILING_DIR = os.path.join(RESULTS_ROOT, 'profiling')
os.makedirs(PROFILING_DIR, exist_ok=True)


def load_fooling_rates(attack='patch_fool'):
    """
    최신 로그에서 특정 공격의 P별 Overall Fooling Rate 파싱.
    SUMMARY 테이블 형식:
      patch_fool     34.70%    78.82%    98.28%
    """
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'result_*.txt')))
    if not files:
        print("  [Warning] No result_*.txt found")
        return {}, []
    latest = files[-1]
    print(f"  Fooling Rate source: {os.path.basename(latest)}")

    with open(latest, 'r', encoding='utf-8') as f:
        content = f.read()

    # 헤더에서 P값 추출
    header = re.search(r'Attack\s+((?:P=\d+\s*)+)', content)
    if not header:
        return {}, []
    p_values = [int(x) for x in re.findall(r'P=(\d+)', header.group(1))]

    # 공격 행 파싱
    for line in content.splitlines():
        m = re.match(rf'\s*{re.escape(attack)}\s+([\d.% ]+)', line)
        if m:
            vals = re.findall(r'([\d.]+)%', m.group(1))
            if len(vals) == len(p_values):
                return {p: float(v) for p, v in zip(p_values, vals)}, p_values

    return {}, p_values


def measure_flops(model):
    try:
        from torchinfo import summary
        dummy  = torch.randn(1, 3, 224, 224)
        result = summary(model, input_data=dummy, verbose=0)
        return result.total_mult_adds, 'measured'
    except ImportError:
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        N     = (224 // patch_size) ** 2 + 1
        D     = model.embed_dim
        L     = len(model.blocks)
        flops = 4 * N * N * D * L
        return flops, 'theoretical'


def measure_latency(model, device):
    dummy = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
    model.eval()

    # CUDA sync
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(dummy)
        sync()

    times = []
    with torch.no_grad():
        for _ in range(MEASURE_RUNS):
            sync()
            t0 = time.perf_counter()
            model(dummy)
            sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    t = torch.tensor(times)
    return float(t.mean()), float(t.std())


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    device = get_device()

    # PatchFool Fooling Rate 로드 (트레이드오프 기준 공격)
    fr_map, _ = load_fooling_rates(attack='patch_fool')
    print(f"  PatchFool Fooling Rates: {fr_map}")
    print(f"\nDevice: {device} | Warmup: {WARMUP_RUNS} | Measure: {MEASURE_RUNS}\n")

    results = {}

    for P in PATCH_SIZES:
        print(f"{'='*50}")
        print(f"  P={P}")
        print(f"{'='*50}")

        model = load_vit_model(P, device)
        if model is None:
            continue
        model.eval()

        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        tokens = (224 // patch_size) ** 2

        params            = count_params(model)
        flops, flops_type = measure_flops(model)
        lat_mean, lat_std = measure_latency(model, device)
        fool_rate         = fr_map.get(P)

        results[P] = {
            'tokens':     tokens,
            'params_M':   params,
            'flops':      flops,
            'flops_type': flops_type,
            'lat_mean':   lat_mean,
            'lat_std':    lat_std,
            'fool_rate':  fool_rate,
        }

        print(f"  Tokens    : {tokens}")
        print(f"  Params    : {params:.1f}M")
        print(f"  FLOPs     : {flops:,.0f}  ({flops_type})")
        print(f"  Latency   : {lat_mean:.2f} ± {lat_std:.2f} ms")
        if fool_rate is not None:
            print(f"  PatchFool FR: {fool_rate:.2f}%")
        print()

    # ── 트레이드오프 요약 표 ─────────────────────────────────────────────────
    valid = {P: r for P, r in results.items() if r}
    if not valid:
        return

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  Security-Efficiency Trade-off Summary  (PatchFool FR)")
    lines.append(f"{'='*70}")

    # 헤더
    p_cols = sorted(valid.keys())
    col_w  = 12
    hdr    = f"  {'Metric':<24}" + "".join(f"{'P='+str(p):>{col_w}}" for p in p_cols)
    lines.append(hdr)
    lines.append(f"  {'-'*68}")

    def row(label, vals):
        return f"  {label:<24}" + "".join(f"{v:>{col_w}}" for v in vals)

    lines.append(row('Tokens',
        [str(valid[p]['tokens']) for p in p_cols]))
    lines.append(row('Params (M)',
        [f"{valid[p]['params_M']:.1f}" for p in p_cols]))
    lines.append(row('FLOPs',
        [f"{valid[p]['flops']/1e9:.2f}G" for p in p_cols]))
    lines.append(row('Latency (ms)',
        [f"{valid[p]['lat_mean']:.2f}±{valid[p]['lat_std']:.2f}" for p in p_cols]))
    lines.append(row('PatchFool FR (%)',
        [f"{valid[p]['fool_rate']:.2f}" if valid[p]['fool_rate'] else 'N/A'
         for p in p_cols]))

    # P=8 기준 상대 비율
    if 8 in valid:
        ref = valid[8]
        lines.append(f"\n  Relative to P=8:")
        lines.append(row('  FLOPs ratio',
            ['1.00x'] + [f"{valid[p]['flops']/ref['flops']:.2f}x"
                         for p in p_cols if p != 8]))
        lines.append(row('  Latency ratio',
            ['1.00x'] + [f"{valid[p]['lat_mean']/ref['lat_mean']:.2f}x"
                         for p in p_cols if p != 8]))
        lines.append(row('  FR delta',
            ['0.00%p'] + [f"{valid[p]['fool_rate']-ref['fool_rate']:+.2f}%p"
                          if valid[p]['fool_rate'] and ref['fool_rate'] else 'N/A'
                          for p in p_cols if p != 8]))

    # 결론

    for line in lines:
        print(line)

    # 저장
    prof_path = os.path.join(PROFILING_DIR, 'efficiency_summary.txt')
    with open(prof_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved: {prof_path}")


if __name__ == '__main__':
    main()