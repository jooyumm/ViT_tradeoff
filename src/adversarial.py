import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================================================
# 공격 패치 크기 (이미지 전체 면적 기준 2%)
# 한 변의 길이 = 224 × √0.02 ≈ 31px
# ================================================================
PATCH_RATIO = 0.02


def get_patch_region(image_size=224, patch_ratio=PATCH_RATIO):
    patch_pixels = int(image_size * (patch_ratio ** 0.5))
    center       = image_size // 2
    start        = center - patch_pixels // 2
    end          = start + patch_pixels
    return start, end


def pgd_patch_attack(model, images, device,
                     steps=40, alpha=2/255,
                     patch_ratio=PATCH_RATIO):
    
    # 2% 패치 공격, 이미지 중앙 고정
    model.eval()
    images = images.clone().detach().to(device)
    start, end = get_patch_region(image_size=images.size(-1),
                                  patch_ratio=patch_ratio)

    # 패치 초기화: 0 근처 작은 랜덤값
    patch = torch.zeros(
        (images.size(0), 3, end - start, end - start), device=device
    )
    patch.requires_grad_(True)

    best_patch = patch.clone().detach()
    max_loss   = torch.full((images.size(0),), -float('inf'), device=device)

    for step in range(steps):
        adv_images = images.clone()
        adv_images[:, :, start:end, start:end] = (
            adv_images[:, :, start:end, start:end] + patch
        ).clamp(0, 1)

        outputs = model(adv_images)

        preds   = outputs.argmax(dim=1).detach()
        loss    = F.cross_entropy(outputs, preds)

        # Best Tracking: 샘플별 최고 loss 패치 저장
        with torch.no_grad():
            loss_per_sample = F.cross_entropy(
                outputs, preds, reduction='none').detach()
            mask             = loss_per_sample > max_loss
            max_loss[mask]   = loss_per_sample[mask]
            best_patch[mask] = patch[mask].detach()

        if patch.grad is not None:
            patch.grad.zero_()
        loss.backward()

        # Gradient Ascent (loss 최대화 방향)
        with torch.no_grad():
            patch.data += alpha * patch.grad.sign()
            patch.data  = patch.data.clamp(-0.5, 0.5)

    # 최적 패치 적용
    adv_images = images.clone()
    with torch.no_grad():
        adv_images[:, :, start:end, start:end] = (
            adv_images[:, :, start:end, start:end] + best_patch
        ).clamp(0, 1)

    return adv_images.detach()