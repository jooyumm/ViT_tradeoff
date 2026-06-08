import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import re
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model
from src.adversarial import pgd_patch_attack

# ================================================================
# 설정
# ================================================================
SEED        = 42
BATCH_SIZE  = 32
NUM_SAMPLES = 10000
DATA_ROOT   = os.path.join(ROOT, 'data')
LOG_DIR     = os.path.join(ROOT, 'results', 'logs')
FIG_DIR     = os.path.join(ROOT, 'results', 'figures', 'outlier_analysis')

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(FIG_DIR, exist_ok=True)


def load_batch_flip_rates(p_size):
    """
    가장 최근 result_*.txt에서 P별 배치 Flip Rate 읽기
    반환: {batch_num: flip_rate} 딕셔너리
    """
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'result_*.txt')))
    if not files:
        print("  [Warning] result_*.txt 없음")
        return {}

    latest = files[-1]
    print(f"  Log source: {latest}")

    with open(latest, 'r', encoding='utf-8') as f:
        content = f.read()

    # P별 섹션 추출
    pattern = rf"P={p_size} \| Tokens.*?(?=P=\d+ \| Tokens|Summary|$)"
    section = re.search(pattern, content, re.DOTALL)
    if not section:
        return {}

    rates = re.findall(r'Batch\s+(\d+)/\d+.*?Flip Rate:\s*([\d.]+)%',
                       section.group())
    return {int(b): float(r) for b, r in rates}


def get_target_batches(flip_dict, top_n=2):
    """
    Flip Rate 상위 top_n (best) + 하위 top_n (worst) 배치 반환
    best: Flip Rate 높음 (공격 성공)
    worst: Flip Rate 낮음 (공격 실패)
    """
    sorted_rates = sorted(flip_dict.items(), key=lambda x: x[1], reverse=True)
    best  = sorted_rates[:top_n]   # Flip Rate 높은 배치
    worst = sorted_rates[-top_n:]  # Flip Rate 낮은 배치
    return best, worst


def get_dataloader():
    val_dir   = os.path.join(DATA_ROOT, 'tiny-imagenet-200', 'val')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    subset  = torch.utils.data.Subset(dataset, range(min(NUM_SAMPLES, len(dataset))))
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False), dataset


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def save_batch_images(loader, dataset, model, device,
                      batch_targets, save_subdir, label):
    """
    batch_targets: [(batch_num, flip_rate), ...]
    save_subdir: 'P16_best', 'P16_worst' 등
    """
    target_dict = {b: r for b, r in batch_targets}
    target_set  = set(target_dict.keys())
    class_names = dataset.classes
    model.eval()

    out_dir = os.path.join(FIG_DIR, save_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[{save_subdir}] 추출 배치: "
          + ", ".join([f"Batch {b} ({r:.1f}%)" for b, r in batch_targets]))

    for batch_idx, (images, labels) in enumerate(loader):
        batch_num = batch_idx + 1
        if batch_num not in target_set:
            continue

        images    = images.to(device)
        flip_rate = target_dict[batch_num]

        with torch.no_grad():
            preds_clean = model(images).argmax(dim=1).cpu().tolist()

        adv_images = pgd_patch_attack(model, images, device)

        with torch.no_grad():
            preds_adv = model(adv_images).argmax(dim=1).cpu().tolist()

        n    = len(images)
        cols = 8
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
        fig.suptitle(
            f'{label} | Batch {batch_num} | Flip Rate: {flip_rate:.2f}%\n'
            f'RED = attacked (prediction changed) | GREEN = defended (prediction held)',
            fontsize=12, fontweight='bold'
        )

        for i in range(n):
            ax     = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
            img_np = denormalize(images[i].cpu()).permute(1, 2, 0).numpy()
            ax.imshow(img_np)

            flipped = (preds_clean[i] != preds_adv[i])
            color   = 'red' if flipped else 'green'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)

            true_idx  = labels[i].item()
            true_name = class_names[true_idx] if true_idx < len(class_names) \
                        else f'cls_{true_idx}'
            title     = f'{true_name}\n(FLIPPED)' if flipped else true_name
            ax.set_title(title, fontsize=6, color=color)
            ax.axis('off')

        for j in range(n, rows * cols):
            ax = axes[j // cols][j % cols] if rows > 1 else axes[j % cols]
            ax.axis('off')

        red_patch   = mpatches.Patch(color='red',   label='Attacked (flipped)')
        green_patch = mpatches.Patch(color='green', label='Defended (held)')
        fig.legend(handles=[red_patch, green_patch],
                   loc='lower center', ncol=2, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))

        fname = os.path.join(out_dir, f'batch{batch_num:02d}_FR{flip_rate:.0f}pct.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

        if batch_num >= max(target_set):
            break


def analyze_model(model, loader, dataset, device, p_size):
    flip_dict = load_batch_flip_rates(p_size)
    if not flip_dict:
        print(f"  P={p_size} Flip Rate 데이터 없음")
        return

    best, worst = get_target_batches(flip_dict, top_n=2)
    print(f"\n  P={p_size} Best (Flip Rate 높음): "
          + str([(b, f"{r:.1f}%") for b, r in best]))
    print(f"  P={p_size} Worst (Flip Rate 낮음): "
          + str([(b, f"{r:.1f}%") for b, r in worst]))

    save_batch_images(loader, dataset, model, device,
                      best,  f'P{p_size}_best',  f'P={p_size}')
    save_batch_images(loader, dataset, model, device,
                      worst, f'P{p_size}_worst', f'P={p_size}')


def main():
    device = get_device()

    for p_size in [16, 32]:
        print(f"\n{'='*50}")
        print(f"  P={p_size} 분석")
        print(f"{'='*50}")

        model = load_vit_model(p_size, device)
        if model is None:
            continue

        loader, dataset = get_dataloader()
        analyze_model(model, loader, dataset, device, p_size)

    print(f"\nDone. Check: {FIG_DIR}")
    print("폴더 구조:")
    print("  outlier_analysis/")
    print("  ├── P16_best/   (Flip Rate 높은 배치 2개)")
    print("  ├── P16_worst/  (Flip Rate 낮은 배치 2개)")
    print("  ├── P32_best/   (Flip Rate 높은 배치 2개)")
    print("  └── P32_worst/  (Flip Rate 낮은 배치 2개)")


if __name__ == '__main__':
    main()