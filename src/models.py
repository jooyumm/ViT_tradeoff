import torch
import timm

# ================================================================
# P별 timm 모델명 후보 (우선순위 순)
# ================================================================
MODEL_FALLBACKS = {
    8: [
        'vit_base_patch8_224.augreg2_in21k_ft_in1k',
        'vit_base_patch8_224.augreg_in21k_ft_in1k',
        'vit_base_patch8_224',
    ],
    16: [
        'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        'vit_base_patch16_224.augreg_in1k',
        'vit_base_patch16_224',
    ],
    32: [
        'vit_base_patch32_224.augreg_in1k',
        'vit_base_patch32_224.augreg2_in21k_ft_in1k',
        'vit_base_patch32_224',
    ],
}


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    return device


def load_vit_model(patch_size, device):
    """
    timm으로 ViT-Base 모델 로드. 후보 모델명을 순서대로 시도.
    로드 후 실제 patch_size가 요청값과 일치하는지 검증.
    """
    candidates = MODEL_FALLBACKS.get(patch_size, [f'vit_base_patch{patch_size}_224'])

    for name in candidates:
        try:
            print(f"  시도: {name}")
            model = timm.create_model(name, pretrained=True)

            # patch_size 검증 — timm 모델이 요청한 P와 실제로 다를 경우 스킵
            actual_p = getattr(model.patch_embed, 'patch_size', None)
            if actual_p is not None:
                # timm은 (H, W) tuple 또는 int 둘 다 사용
                actual_p = actual_p[0] if isinstance(actual_p, (tuple, list)) else actual_p
                if actual_p != patch_size:
                    print(f"  ✗ patch_size 불일치: 요청={patch_size}, 실제={actual_p} → 스킵")
                    continue

            model = model.to(device)
            model.eval()
            num_tokens = (224 // patch_size) ** 2
            print(f"  ✓ 로드 성공: {name}  "
                  f"(P={patch_size}, tokens={num_tokens}, "
                  f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)")
            return model

        except Exception as e:
            print(f"  ✗ 실패: {e}")

    print(f"  모든 후보 실패 → P={patch_size} 스킵")
    return None


def get_model_info(model, patch_size, img_size=224):
    """
    모델 기본 정보 반환 (로그 / 결과 테이블용).

    Returns
    -------
    dict with keys:
      patch_size  : int
      num_tokens  : int   (패치 수, CLS 제외)
      num_params  : float (단위: M)
      embed_dim   : int
    """
    num_tokens = (img_size // patch_size) ** 2
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    embed_dim  = model.embed_dim if hasattr(model, 'embed_dim') else -1

    return {
        'patch_size': patch_size,
        'num_tokens': num_tokens,
        'num_params': num_params,
        'embed_dim':  embed_dim,
    }