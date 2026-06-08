"""
main.py — ViT 적대적 패치 공격 실험 실행기

사용법:
  python experiments/main.py --attacks pgd lavan patch_fool --patch_sizes 8 16 32
  python experiments/main.py --attacks patch_fool --patch_sizes 8 16 32 --num_samples 5000 --pf_iters 250

지표:
  CA  (Clean Accuracy)      = clean_correct / total
  RA  (Robust Accuracy)     = correct_after_attack / total     Higher is Better
  ASR (Attack Success Rate) = FT / clean_correct               Lower is Better
  Δ   (Accuracy Drop)       = CA - RA
"""

import argparse
import torch
import numpy as np
import os
import sys
from datetime import datetime
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from experiments.attack import run_attack_on_batch, ATTACK_DEFAULTS


def get_args():
    parser = argparse.ArgumentParser(description='ViT Adversarial Patch Experiment')

    parser.add_argument('--attacks', nargs='+', default=['patch_fool'],
                        choices=['patch_fool', 'pgd', 'fgsm', 'lavan'])
    parser.add_argument('--patch_sizes', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--seed',        type=int, default=42)

    # patch_fool
    parser.add_argument('--pf_attack_mode',  default='Attention',
                        choices=['Attention', 'CE_loss'])
    parser.add_argument('--pf_iters',        type=int,   default=250)
    parser.add_argument('--pf_num_patch',    type=int,   default=1)
    parser.add_argument('--pf_patch_select', default='Attn', choices=['Attn', 'Rand'])
    parser.add_argument('--pf_mild_l_inf',   type=float, default=0.0)

    # pgd
    parser.add_argument('--pgd_epsilon', type=float, default=8/255)
    parser.add_argument('--pgd_alpha',   type=float, default=2/255)
    parser.add_argument('--pgd_steps',   type=int,   default=40)

    # lavan
    parser.add_argument('--lavan_patch_ratio', type=float, default=0.02)
    parser.add_argument('--lavan_steps',       type=int,   default=40)
    parser.add_argument('--lavan_alpha',       type=float, default=2/255)

    parser.add_argument('--data_root', default=os.path.join(ROOT, 'data'))
    parser.add_argument('--log_dir',   default=os.path.join(ROOT, 'results', 'logs'))

    return parser.parse_args()


class Tee:
    def __init__(self, file):
        self.file = file; self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data); self.file.write(data)
    def flush(self):
        self.stdout.flush(); self.file.flush()


def build_attack_kwargs(args, attack_name):
    if attack_name == 'patch_fool':
        return {
            'attack_mode':        args.pf_attack_mode,
            'train_attack_iters': args.pf_iters,
            'num_patch':          args.pf_num_patch,
            'patch_select':       args.pf_patch_select,
            'mild_l_inf':         args.pf_mild_l_inf,
        }
    elif attack_name == 'pgd':
        return {'epsilon': args.pgd_epsilon, 'alpha': args.pgd_alpha,
                'steps':   args.pgd_steps}
    elif attack_name == 'lavan':
        return {'patch_ratio': args.lavan_patch_ratio,
                'steps':       args.lavan_steps,
                'alpha':       args.lavan_alpha}
    return {}


def run_experiment(model, loader, device, attack_name, patch_size_model,
                   num_batches, attack_kwargs):
    model.eval()
    # 누적 카운터
    total_FF = total_FT = total_TF = total_TT = 0
    total_valid = 0
    batch_asrs  = []

    batch_iter = iter(loader)
    for b_idx in range(num_batches):
        try:
            images, labels = next(batch_iter)
        except StopIteration:
            break

        metrics, _ = run_attack_on_batch(
            model, images, labels, device,
            attack_name=attack_name,
            patch_size_model=patch_size_model,
            attack_kwargs=attack_kwargs,
        )

        total_FF    += metrics['FF']
        total_FT    += metrics['FT']
        total_TF    += metrics['TF']
        total_TT    += metrics['TT']
        total_valid += metrics['valid']
        batch_asrs.append(metrics['ASR'])

        print(f"    Batch {b_idx+1:2d}/{num_batches} | "
              f"CA:{metrics['CA']:5.1f}% "
              f"RA:{metrics['RA']:5.1f}% "
              f"ASR:{metrics['ASR']:5.1f}% "
              f"[FF:{metrics['FF']} FT:{metrics['FT']} "
              f"TF:{metrics['TF']} TT:{metrics['TT']}]")

    # 전체 누적 지표
    clean_correct = total_FF + total_FT
    adv_correct   = total_FF + total_TF

    overall = {
        'CA':       100.0 * clean_correct / total_valid if total_valid > 0 else 0.0,
        'RA':       100.0 * adv_correct   / total_valid if total_valid > 0 else 0.0,
        'ASR':      100.0 * total_FT / clean_correct    if clean_correct > 0 else 0.0,
        'FF':       total_FF,
        'FT':       total_FT,
        'TF':       total_TF,
        'TT':       total_TT,
        'valid':    total_valid,
    }
    overall['acc_drop'] = overall['CA'] - overall['RA']
    return overall


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path    = os.path.join(args.log_dir, f'result_{timestamp}.txt')
    num_batches = args.num_samples // args.batch_size
    device      = get_device()
    orig_stdout = sys.stdout

    with open(log_path, 'w', encoding='utf-8') as f:
        sys.stdout = Tee(f)
        try:
            print("=" * 65)
            print("  ViT Adversarial Robustness Experiment")
            print("=" * 65)
            print(f"  Device      : {device}")
            print(f"  Attacks     : {args.attacks}")
            print(f"  Patch sizes : P = {args.patch_sizes}")
            print(f"  Samples     : {num_batches * args.batch_size} "
                  f"({args.batch_size} x {num_batches} batches)")
            print(f"  Metrics     : CA / RA / ASR")
            print(f"    CA  = Clean Accuracy       = clean_correct / total")
            print(f"    RA  = Robust Accuracy      = correct_after_attack / total")
            print(f"    ASR = Attack Success Rate  = FT / clean_correct")
            print(f"    Δ   = CA - RA  (accuracy drop due to attack)")
            print()

            loader, _ = get_dataloader(
                batch_size=args.batch_size,
                num_samples=num_batches * args.batch_size,
                data_root=args.data_root,
            )

            # results[attack][P] = overall dict
            results = {atk: {} for atk in args.attacks}

            for P, attack_name in product(args.patch_sizes, args.attacks):
                print(f"\n{'─'*65}")
                print(f"  Attack: {attack_name:12s} | P={P} | "
                      f"Tokens: {(224 // P)**2}")
                print(f"{'─'*65}")

                model = load_vit_model(P, device)
                if model is None:
                    print(f"  [SKIP] P={P} 모델 로드 실패")
                    results[attack_name][P] = None
                    continue

                attack_kwargs = build_attack_kwargs(args, attack_name)
                overall = run_experiment(
                    model, loader, device,
                    attack_name=attack_name,
                    patch_size_model=P,
                    num_batches=num_batches,
                    attack_kwargs=attack_kwargs,
                )
                results[attack_name][P] = overall

                print(f"\n  ── Overall ──────────────────────────────────────")
                print(f"  CA  (Clean Accuracy)   : {overall['CA']:.2f}%")
                print(f"  RA  (Robust Accuracy)  : {overall['RA']:.2f}%")
                print(f"  ASR (Attack Success)   : {overall['ASR']:.2f}%")
                print(f"  Δ   (CA - RA)          : {overall['acc_drop']:.2f}%p")
                print(f"  Sample breakdown       : "
                      f"FF={overall['FF']} FT={overall['FT']} "
                      f"TF={overall['TF']} TT={overall['TT']} "
                      f"(valid={overall['valid']})")

            # ── SUMMARY ──────────────────────────────────────────────────
            print(f"\n\n{'='*65}")
            print(f"  SUMMARY")
            print(f"{'='*65}")

            P_list = args.patch_sizes
            col_w  = 10

            for metric in ['CA', 'RA', 'ASR', 'acc_drop']:
                label = {'CA':'Clean Acc','RA':'Robust Acc',
                         'ASR':'ASR','acc_drop':'Δ (CA-RA)'}[metric]
                header = f"  {label:<14}" + "".join(
                    f"{'P='+str(P):>{col_w}}" for P in P_list)
                print(header)
                print("  " + "─" * (14 + col_w * len(P_list)))

                for attack_name in args.attacks:
                    row = f"  {attack_name:<14}"
                    for P in P_list:
                        r = results[attack_name].get(P)
                        if r is None:
                            row += f"{'N/A':>{col_w}}"
                        else:
                            row += f"{r[metric]:>{col_w-1}.2f}%"
                    print(row)
                print()

        finally:
            sys.stdout = orig_stdout

    print(f"\nLog saved: {log_path}")


if __name__ == '__main__':
    main()