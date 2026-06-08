"""
test_attacks.py — 공격 1장 시각화 검증
각 공격의 패치 위치/영역이 의도대로인지 확인

사용법:
  python test_attacks.py
  -> results/figures/attack_test_*.png 저장
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.pgd import pgd_attack
from src.attacks.lavan import lavan_attack
from src.attacks.patch_fool import patch_fool_attack

OUT_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

MU  = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denorm(t):
    """정규화 해제 → [0,1] numpy HWC"""
    x = t.cpu().numpy().transpose(1,2,0)
    return np.clip(x * STD + MU, 0, 1)

def diff_heatmap(orig, adv):
    """픽셀별 변화량 heatmap (정규화 공간)"""
    d = (adv - orig).abs().sum(dim=0).cpu().numpy()
    return d / (d.max() + 1e-8)

def test_one_image(P=16):
    device = get_device()
    model  = load_vit_model(P, device)
    model.eval()

    loader, _ = get_dataloader(batch_size=1, num_samples=16,
                               data_root=os.path.join(ROOT, 'data'))

    # 정분류 샘플 1장 찾기
    img, lbl = None, None
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            pred = model(images).argmax(dim=1)
        if pred[0] == labels[0]:
            img, lbl = images[0:1], labels[0:1]
            break

    if img is None:
        print("정분류 샘플 없음"); return

    print(f"P={P} | label={lbl.item()} | image shape={img.shape}")

    attacks = {
        'PGD':       lambda: pgd_attack(model, img, lbl, device,
                         epsilon=8/255, alpha=2/255, steps=40, random_start=True),
        'LaVAN':     lambda: lavan_attack(model, img, lbl, device,
                         patch_ratio=0.02, steps=40, alpha=2/255),
        'PatchFool': lambda: patch_fool_attack(model, img, lbl, device,
                         patch_size_model=P,
                         attack_mode='CE_loss',
                         train_attack_iters=50,
                         num_patch=1,
                         patch_select='Attn')[0],
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Attack Visualization (P={P}, label={lbl.item()})',
                 fontsize=14, fontweight='bold')

    orig_np = denorm(img[0])

    for row, (name, atk_fn) in enumerate(attacks.items()):
        print(f"  {name} 실행 중...")
        adv = atk_fn()

        with torch.no_grad():
            pred_clean = model(img).argmax(dim=1).item()
            pred_adv   = model(adv).argmax(dim=1).item()

        adv_np   = denorm(adv[0])
        heat     = diff_heatmap(img[0], adv[0])
        diff_np  = np.clip((adv_np - orig_np) * 5 + 0.5, 0, 1)

        # col 0: 원본
        axes[row,0].imshow(orig_np)
        axes[row,0].set_title(f'Original\npred: {pred_clean}')
        axes[row,0].axis('off')

        # col 1: 공격 이미지
        axes[row,1].imshow(adv_np)
        axes[row,1].set_title(f'{name}\npred: {pred_adv} '
                              f'{"✓ FOOL" if pred_adv != pred_clean else "✗ fail"}')
        axes[row,1].axis('off')

        # col 2: 차이 heatmap (패치 위치 확인 핵심)
        axes[row,2].imshow(orig_np, alpha=0.4)
        axes[row,2].imshow(heat, cmap='hot', alpha=0.7)
        axes[row,2].set_title(f'Perturbation Heatmap\n(밝을수록 변화 큼)')
        axes[row,2].axis('off')

        # col 3: 픽셀 차이 (amplified)
        axes[row,3].imshow(diff_np)
        axes[row,3].set_title(f'Diff x5\n(변화 영역 확인)')
        axes[row,3].axis('off')

        # 변화 픽셀 비율 출력
        changed = (heat > 0.01).sum()
        total   = heat.size
        print(f"    pred: {pred_clean} -> {pred_adv} | "
              f"변화 픽셀: {changed}/{total} ({100*changed/total:.1f}%)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'attack_test_P{P}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  저장: {out}")

if __name__ == '__main__':
    for P in [16, 32]:
        print(f"\n{'='*50}")
        print(f"P={P} 테스트")
        print('='*50)
        test_one_image(P)