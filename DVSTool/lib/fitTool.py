import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable


def backWarp(img: torch.Tensor, flow: torch.Tensor):
    device = img.device
    N, C, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False).to(device)
    gridY = torch.tensor(gridY, requires_grad=False).to(device)

    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v

    # range -1 to 1
    x = 2 * x / (W - 1.0) - 1.0
    y = 2 * y / (H - 1.0) - 1.0
    # stacking X and Y
    grid = torch.stack((x, y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')

    # mask = torch.ones_like(img, requires_grad=False)
    # mask = F.grid_sample(mask, grid)
    #
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1

    # return imgOut * (mask.detach())
    return imgOut


class IdCoRe(object):
    def __init__(self, intime, device, target='idx'):
        self.device = device
        if isinstance(intime, int):
            intime = torch.tensor(intime).to(self.device)

        if target == 'idx':
            self.coord = intime + 1  # if the index in frameT is 0
        elif target == 'coord':
            self.coord = intime  # then the related time in dct coord is 1
        elif target == 'time':  # and the real time is 0.125s
            self.coord = intime * 8.0

    @property
    def idx(self):
        return (self.coord - 1).int()

    @idx.setter
    def idx(self, x):
        x = torch.tensor(x)
        self.coord = x + 1

    @property
    def time(self):
        return self.coord.float() / 8.0

    @time.setter
    def time(self, x):
        x = torch.tensor(x).to(self.device)
        self.coord = x * 8.0


def getAccFlow(aF, bF, aB, bB, t, device):
    F0t = aF * (t ** 2) + bF * t
    F1t = aB * ((1 - t) ** 2) + bB * (1 - t)

    return F0t.to(device), F1t.to(device)


def getAccParam(F0_1, F01):
    a = (F01 + F0_1) / 2.0
    b = (F01 - F0_1) / 2.0

    return a, b
