"""
attack.py — 공격 실행 + 지표 계산 통합

지표 정의:
  CA  (Clean Accuracy)    = clean_correct / total
  RA  (Robust Accuracy)   = correct_after_attack / total
  ASR (Attack Success Rate) = FT / (FT + FF) = FT / clean_correct

샘플별 4가지 케이스:
  FF: clean 맞음 → attack 맞음  (강건한 샘플)
  FT: clean 맞음 → attack 틀림  (공격 성공 — ASR 분자)
  TF: clean 틀림 → attack 맞음  (노이즈, 제외)
  TT: clean 틀림 → attack 틀림  (공격 무관하게 틀림)
"""

import torch
from src.attacks.patch_fool import patch_fool_attack
from src.attacks.pgd import pgd_attack
from src.attacks.fgsm import fgsm_attack
from src.attacks.lavan import lavan_attack

ATTACK_DEFAULTS = {
    'patch_fool': {
        'num_patch':          1,
        'attack_mode':        'Attention',
        'patch_select':       'Attn',
        'attn_layer_idx':     4,
        'train_attack_iters': 250,
        'attack_lr':          0.22,
        'step_size':          10,
        'gamma':              0.95,
        'atten_loss_weight':  0.002,
        'mild_l_inf':         0.0,
    },
    'pgd': {
        'epsilon':      8 / 255,
        'alpha':        2 / 255,
        'steps':        40,
        'random_start': True,
    },
    'fgsm': {
        'epsilon': 8 / 255,
    },
    'lavan': {
        'patch_ratio': 0.02,
        'steps':       40,
        'alpha':       2 / 255,
    },
}


def compute_metrics(preds_clean, preds_adv, labels):
    """
    샘플별 4가지 케이스 분류 후 CA / RA / ASR 계산.

    Parameters
    ----------
    preds_clean : list[int]  공격 전 예측
    preds_adv   : list[int]  공격 후 예측
    labels      : list[int]  정답 레이블 (-1이면 remapping 실패 샘플)

    Returns
    -------
    dict with keys:
      total, valid           : 전체 샘플 수, 유효 샘플 수 (-1 제외)
      FF, FT, TF, TT         : 케이스별 샘플 수
      CA, RA, ASR            : 각 지표 (%)
      acc_drop               : CA - RA (공격으로 인한 정확도 손실)
    """
    total = len(labels)

    FF = FT = TF = TT = 0
    valid = 0

    for pc, pa, l in zip(preds_clean, preds_adv, labels):
        if l == -1:   # remapping 실패 샘플 제외
            continue
        valid += 1
        clean_correct = (pc == l)
        adv_correct   = (pa == l)

        if clean_correct and adv_correct:
            FF += 1
        elif clean_correct and not adv_correct:
            FT += 1
        elif not clean_correct and adv_correct:
            TF += 1
        else:
            TT += 1

    clean_correct_count = FF + FT   # 공격 전 맞은 샘플
    adv_correct_count   = FF + TF   # 공격 후 맞은 샘플

    CA  = 100.0 * clean_correct_count / valid if valid > 0 else 0.0
    RA  = 100.0 * adv_correct_count   / valid if valid > 0 else 0.0
    ASR = 100.0 * FT / clean_correct_count if clean_correct_count > 0 else 0.0

    return {
        'total':    total,
        'valid':    valid,
        'FF':       FF,
        'FT':       FT,
        'TF':       TF,
        'TT':       TT,
        'CA':       CA,
        'RA':       RA,
        'ASR':      ASR,
        'acc_drop': CA - RA,
    }


def run_attack_on_batch(model, images, labels, device,
                        attack_name: str,
                        patch_size_model: int = 16,
                        attack_kwargs: dict = None):
    """
    배치 하나에 대해 공격 실행 → 지표 반환.

    Returns
    -------
    metrics    : dict  (CA, RA, ASR, FF, FT, TF, TT, ...)
    adv_images : Tensor
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    kwargs = dict(ATTACK_DEFAULTS.get(attack_name, {}))
    if attack_kwargs:
        kwargs.update(attack_kwargs)

    with torch.no_grad():
        preds_clean = model(images).argmax(dim=1).cpu().tolist()

    if attack_name == 'patch_fool':
        adv_images, _ = patch_fool_attack(
            model, images, labels, device,
            patch_size_model=patch_size_model, **kwargs)
    elif attack_name == 'pgd':
        adv_images = pgd_attack(model, images, labels, device, **kwargs)
    elif attack_name == 'fgsm':
        adv_images = fgsm_attack(model, images, labels, device, **kwargs)
    elif attack_name == 'lavan':
        adv_images = lavan_attack(model, images, labels, device, **kwargs)
    else:
        raise ValueError(f"Unknown attack: '{attack_name}'")

    with torch.no_grad():
        preds_adv = model(adv_images).argmax(dim=1).cpu().tolist()

    labels_list = labels.cpu().tolist()
    metrics     = compute_metrics(preds_clean, preds_adv, labels_list)

    return metrics, adv_images