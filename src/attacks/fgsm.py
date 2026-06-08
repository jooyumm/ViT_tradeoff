"""
FGSM Attack - Goodfellow et al., 2014
입력이 정규화된 텐서임을 고려.
"""

import torch
import torch.nn.functional as F

_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def fgsm_attack(
    model,
    images,
    labels,
    device,
    epsilon=8/255,
):
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    std     = _STD.to(device)
    eps_t   = epsilon / std      # 정규화 공간 epsilon

    images.requires_grad_(True)
    out  = model(images)
    loss = F.cross_entropy(out, labels)
    loss.backward()

    adv = (images + eps_t * images.grad.sign()).detach()
    return adv