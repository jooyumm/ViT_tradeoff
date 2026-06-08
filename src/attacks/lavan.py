import torch
import torch.nn.functional as F

_MU  = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def lavan_attack(
    model,
    images,
    labels,
    device,
    patch_ratio=0.02,
    steps=40,
    alpha=2/255,
    targeted=False,
    target_labels=None,
):
    """
    LaVAN: Localized and Visible Adversarial Noise (Karmon et al., ICML 2018)

    패치 위치: 공격 시작 시 랜덤 선택 후 고정 (논문 기본 설정)
    - 논문: "location is randomly chosen once at the beginning of the attack"
    - 패치가 이미지 경계를 벗어나지 않도록 clamp
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    B, C, H, W = images.shape

    mu  = _MU.to(device)
    std = _STD.to(device)

    # 패치 크기 계산 (면적 기준 patch_ratio)
    patch_px = int(H * (patch_ratio ** 0.5))

    # 패치 위치 랜덤 선택 (배치 내 모든 샘플 동일 위치 — 논문 기본)
    max_start = H - patch_px
    s_h = torch.randint(0, max(max_start, 1), (1,)).item()
    s_w = torch.randint(0, max(max_start, 1), (1,)).item()
    e_h = s_h + patch_px
    e_w = s_w + patch_px

    # 정규화 공간 alpha
    alpha_t = alpha / std.to(device)

    # perturbation 초기화 (패치 영역만)
    delta = torch.zeros_like(images)

    for _ in range(steps):
        delta.requires_grad_(True)
        out = model(images + delta)

        if targeted:
            t    = target_labels.to(device)
            loss = -F.cross_entropy(out, t)
        else:
            loss = F.cross_entropy(out, labels)

        loss.backward()

        with torch.no_grad():
            grad_sign = delta.grad.detach().sign()
            new_delta  = delta.detach().clone()
            # 패치 영역만 업데이트
            new_delta[:, :, s_h:e_h, s_w:e_w] += \
                alpha_t * grad_sign[:, :, s_h:e_h, s_w:e_w]
            # 정규화 공간 [-mu/std, (1-mu)/std] clamp
            lo = (-mu / std).to(device)
            hi = ((1 - mu) / std).to(device)
            new_delta[:, :, s_h:e_h, s_w:e_w] = torch.max(
                torch.min(
                    new_delta[:, :, s_h:e_h, s_w:e_w],
                    hi - images[:, :, s_h:e_h, s_w:e_w]
                ),
                lo - images[:, :, s_h:e_h, s_w:e_w]
            )
            # 패치 외부 영역 0으로 유지
            new_delta[:, :, :s_h, :]  = 0
            new_delta[:, :, e_h:, :]  = 0
            new_delta[:, :, :, :s_w]  = 0
            new_delta[:, :, :, e_w:]  = 0
            delta = new_delta

    return (images + delta).detach()