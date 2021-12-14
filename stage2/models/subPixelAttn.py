import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.warp import backWarp as bWarp
# from lib.softSplit import ModuleSoftsplat
from models.baseModule import Correlation


class MultiScaleAttn(nn.Module):
    def __init__(self):
        super(MultiScaleAttn, self).__init__()
        # self.p = [8, 4, 2]
        self.p = [8, 4, 2]
        self.norm = F.normalize
        self.ConvIn = nn.Conv2d(in_channels=1152, out_channels=32, kernel_size=1, bias=False)
        self.corr4x = Correlation(pad_size=self.p[0], kernel_size=1, max_displacement=self.p[0], stride1=1, stride2=1)

    def maxId2Offset(self, maxIdx: torch.Tensor, pSize):
        U = maxIdx % (pSize * 2 + 1) - pSize
        V = maxIdx.int() / int((pSize * 2 + 1)) - pSize
        return torch.stack([U.float(), V.float()], dim=1)

    def getSubOffset(self, cosDis, maxIdx, subAttnConv):
        nx, sxsx, hk, wk = cosDis.shape
        sx = int(np.sqrt(sxsx))
        cout = subAttnConv.shape[0]

        maxIdx = maxIdx.reshape(nx, 1, hk, wk).expand(nx, cout, hk, wk).reshape(nx * cout, 1, hk, wk)

        l2Dis: torch.Tensor = 2.0 - 2.0 * cosDis  # N, hkwk, hqwq
        l2Dis = l2Dis.permute(0, 2, 3, 1).reshape(nx, 1, hk * wk, sx, sx)  # N, 1, hkwk, 17, 17

        l2Dis = F.pad(l2Dis, [1, 1, 1, 1], value=2)
        ABC = F.conv3d(input=l2Dis, weight=subAttnConv, bias=None, stride=(1, 1, 1))  # N, 5, hkwk, 17, 17

        ABC = ABC.reshape(nx * cout, hk, wk, sxsx).contiguous()  # nx * 5, hk, wk, sxsx
        ABC = ABC.permute(0, 3, 1, 2).contiguous()  # nx * 5, sxsx, hk, wk

        ABCHard = torch.gather(ABC, dim=1, index=maxIdx)
        # nx * 5, 1, hk, wk
        ABCHard = ABCHard.reshape(nx, cout, hk, wk)  # nx, 5, hqwq
        #
        subOffU = - ABCHard[:, 2, ...] / (ABCHard[:, 0, ...].clamp(min=1e-6))  # nx, hqwq
        subOffU = subOffU.clamp(max=1, min=-1).reshape(nx, 1, hk, wk)  # nx, 1, hq, wq
        #
        subOffV = - ABCHard[:, 3, ...] / (ABCHard[:, 1, ...].clamp(min=1e-6))  # nx, 1, hqwq
        subOffV = subOffV.clamp(max=1, min=-1).reshape(nx, 1, hk, wk)  # nx, 1, hq, wq
        return torch.cat([subOffU, subOffV], dim=1)

    def forward(self, Kt1x, Kt2x, KVt4x, V01x, V02x, KV04x, subAttnMatC):
        N, C, H, W = KVt4x.shape

        KVt4x_unfold = self.ConvIn(F.unfold(KVt4x, kernel_size=(3, 3), padding=1).reshape(N, -1, H, W))
        KV04x_unfold = self.ConvIn(F.unfold(KV04x, kernel_size=(3, 3), padding=1).reshape(N, -1, H, W))

        cosDis4x = self.corr4x(self.norm(KVt4x_unfold, dim=1),
                               self.norm(KV04x_unfold, dim=1)) * KV04x_unfold.shape[1]  # N, (17*17), Hk, Wk

        maxValue4x, maxIdx4x = torch.max(cosDis4x, dim=1)  # [N, Hq, Wq]
        maxValue4x = maxValue4x.view(N, 1, H, W)

        hardOffset4x = self.maxId2Offset(maxIdx4x, self.p[0])
        subOffset4x = self.getSubOffset(cosDis4x, maxIdx4x, subAttnMatC.detach())
        flowOff4x = hardOffset4x + subOffset4x

        KV04x_unfold = F.unfold(KV04x, kernel_size=(3, 3), padding=1).reshape(N, -1, H, W)
        V02x_unfold = F.unfold(V02x, kernel_size=(6, 6), padding=2, stride=2).reshape(N, -1, H, W)
        V01x_unfold = F.unfold(V01x, kernel_size=(12, 12), padding=4, stride=4).reshape(N, -1, H, W)

        T4x_unfold = bWarp(KV04x_unfold, flowOff4x).reshape(N, -1, H * W)
        T2x_unfold = bWarp(V02x_unfold, flowOff4x).reshape(N, -1, H * W)
        T1x_unfold = bWarp(V01x_unfold, flowOff4x).reshape(N, -1, H * W)

        T4x = F.fold(T4x_unfold, output_size=(H, W), kernel_size=(3, 3), padding=1, stride=1) / (3. * 3.)
        T2x = F.fold(T2x_unfold, output_size=(H * 2, W * 2), kernel_size=(6, 6), padding=2, stride=2) / (3. * 3.)
        T1x = F.fold(T1x_unfold, output_size=(H * 4, W * 4), kernel_size=(12, 12), padding=4, stride=4) / (3. * 3.)

        S = maxValue4x.view(N, 1, H, W)

        return S, T4x, T2x, T1x
