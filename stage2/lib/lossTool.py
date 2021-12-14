import torch
import torch.nn.functional as F
from lib.warp import forwadWarp
import torch.nn as nn

fsplat = forwadWarp(bilinear=False)
fbiwarp = forwadWarp()


def TPerceptualLoss(gt_lv1, gt_lv2, gt_lv3, T_lv1, T_lv2, T_lv3):
    loss_texture = F.mse_loss(gt_lv3.detach(), T_lv3)
    loss_texture += F.mse_loss(gt_lv2.detach(), T_lv2)
    loss_texture += F.mse_loss(gt_lv1.detach(), T_lv1)

    loss_texture /= 3.

    return loss_texture


def l1Loss(source, target, reduction='mean', mask=None):
    if mask is None:
        return F.l1_loss(source, target, reduction=reduction)
    else:
        return F.l1_loss(source * mask, target * mask, reduction=reduction)


def l2Loss(source, target, mask=None, reduction='none'):
    if mask is None:
        return F.mse_loss(source, target, reduction=reduction)
    else:
        return F.mse_loss(source * mask, target * mask, reduction=reduction)


def CharbonnierLoss(source, target, mask=None):
    eps = 1e-6

    if mask is None:
        diff: torch.Tensor = source - target

    else:
        diff: torch.Tensor = (source - target) * mask
    loss = torch.sqrt(diff ** 2 + eps ** 2).mean()
    return loss


def minLoss(I0t: torch.Tensor, I1t: torch.Tensor, It: torch.Tensor):
    I0t.requires_grad_(True)
    I1t.requires_grad_(True)

    loss0t = F.mse_loss(I0t, It.detach(), reduction='none')
    loss1t = F.mse_loss(I1t, It.detach(), reduction='none')

    minLoss = torch.min(loss0t, loss1t).mean()

    return minLoss
