import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baseModule import ConvBlock, BaseNet, Interp, FuseBlock, conv1x1, conv2D3x3, ResBlock
# from models.subPixelAttn import MSAttnSimple, MSAttnFuse
from models.subPixelAttn import MultiScaleAttn
from collections import OrderedDict
import math
from lib.checkTool import checkRefineInput
import numpy as np

baseScale = 32
inCh = 11


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv2D3x3(baseScale * 4, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv2D3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats * 4, n_feats)
        self.conv23 = conv1x1(n_feats * 2, n_feats)
        self.conv_merge = conv2D3x3(n_feats * 3, n_feats)
        self.conv_tail1 = conv2D3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=True)
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge(torch.cat((x3, x13, x23), dim=1)))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        # x = torch.clamp(x, -2, 2)

        return x


class AttExtractor(BaseNet):
    def __init__(self, inCh=inCh, cfg=None):
        super(AttExtractor, self).__init__(cfg.netInitType, cfg.netInitGain)
        self.baseScale = baseScale

        self.slice1 = ConvBlock(in_channels=inCh, out_channels=self.baseScale, kernel_size=3, stride=1, cfg=cfg)
        self.slice2 = nn.Sequential(OrderedDict([
            ('conv1',
             ConvBlock(in_channels=self.baseScale, out_channels=self.baseScale, kernel_size=3, stride=1, cfg=cfg)),
            ('Poolling', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2',
             ConvBlock(in_channels=self.baseScale, out_channels=self.baseScale * 2, kernel_size=3, stride=1, cfg=cfg))
        ]))

        self.slice3 = nn.Sequential(OrderedDict([
            ('conv1',
             ConvBlock(in_channels=self.baseScale * 2, out_channels=self.baseScale * 2, kernel_size=3, stride=1,
                       cfg=cfg)),
            ('Poolling', nn.MaxPool2d(kernel_size=2, stride=2)),
            (
                'conv2',
                ConvBlock(in_channels=self.baseScale * 2, out_channels=self.baseScale * 4, kernel_size=3, stride=1,
                          cfg=cfg))
        ]))

        self.randomInitNet()

    def forward(self, x, isGT=False):
        # prev = torch.is_grad_enabled()
        # if isGT:
        #     torch.set_grad_enabled(False)
        x = self.slice1(x)
        x_lv1 = x  # 32
        x = self.slice2(x)
        x_lv2 = x  # 64
        x = self.slice3(x)
        x_lv3 = x  # 128
        # torch.set_grad_enabled(prev)
        return x_lv1, x_lv2, x_lv3


class AttDecoderFuse(BaseNet):
    def __init__(self, cfg):
        super(AttDecoderFuse, self).__init__(cfg.netInitType, cfg.netInitGain)

        self.SFE = SFE(num_res_blocks=4, n_feats=32, res_scale=1)
        self.ConvIn = nn.Sequential(nn.ReplicationPad2d([1, 1, 1, 1]),
                                    nn.Conv2d(in_channels=288, out_channels=64, kernel_size=3,
                                              stride=1, bias=False)
                                    )
        self.RB0 = nn.ModuleList()

        for i in range(4):
            self.RB0.append(ResBlock(in_channels=64, out_channels=64, res_scale=1))

        self.AADBlk1 = FuseBlock(cin=64, cout=32, c_ef=64)

        self.RB1 = nn.ModuleList()

        for i in range(4):
            self.RB1.append(ResBlock(in_channels=32, out_channels=32, res_scale=1))

        self.AADBlk2 = FuseBlock(cin=32, cout=16, c_ef=32)

        self.RB2 = nn.ModuleList()

        for i in range(2):
            self.RB2.append(ResBlock(in_channels=16, out_channels=16, res_scale=1))

        self.merge_tail = MergeTail(16)
        self.Up2x = Interp(scale=2)
        self.randomInitNet()

        with torch.no_grad():
            self.merge_tail.conv_tail2.weight.data.fill_(0)
            self.merge_tail.conv_tail2.bias.data.fill_(0)
        #     self.merge_Mask.conv_tail2.weight.data.fill_(0)
        #     self.merge_Mask.conv_tail2.bias.data.fill_(5)

    def forward(self, KVt4x, T4x, T2x, T1x, ST4x, ST2x, ST1x):
        ### shallow feature extraction
        KVt4x_ = self.SFE(KVt4x)  # 128->32
        m0 = self.ConvIn(torch.cat([KVt4x_, T4x, ST4x], dim=1))

        m0Res = m0
        for i in range(4):
            m0Res = self.RB0[i](m0Res)
        m0 = m0 + m0Res

        m1 = self.AADBlk1(self.Up2x(m0), T2x, ST2x)

        m1Res = m1
        for i in range(2):
            m1Res = self.RB1[i](m1Res)
        m1 = m1 + m1Res

        m2 = self.AADBlk2(self.Up2x(m1), T1x, ST1x)

        m2Res = m2
        for i in range(2):
            m2Res = self.RB2[i](m2Res)
        m2 = m2 + m2Res

        x = self.merge_tail(m0, m1, m2)
        return x


class refineNet(BaseNet):
    def __init__(self, cfg):
        super(refineNet, self).__init__(cfg.netInitType, cfg.netInitGain)
        self.cfg = cfg
        self.netScale = 64

        self.feaExt = AttExtractor(inCh=inCh, cfg=cfg)
        self.AttnMultiScale = MultiScaleAttn()

        self.subAttnMatC = nn.Parameter(torch.from_numpy(np.load('./matrixC.npy')).
                                        float().reshape(5, 1, 1, 3, 3)[0:4, ...], requires_grad=False)

        self.up2x = Interp(scale=2)
        self.up4x = Interp(scale=4)

        self.AttDecoder = AttDecoderFuse(cfg=cfg)

        if cfg.step in [2, 3]:
            self.initPreweight(cfg.pathWeight)

    def getWeight(self, pathPreWeight: str = None):
        checkpoints = torch.load(pathPreWeight, map_location=torch.device('cpu'))
        try:
            weightDict = checkpoints['attnRefine']
        except Exception as e:
            weightDict = checkpoints['model_state_dict']
        return weightDict

    def forward(self, IE0, IEt, IE1, ST4x, ST2x, ST1x):

        # N, C, H, W = IEt.shape
        ItStage1 = IEt[:, 0:3, ...]
        # for testStage1-------------------------------------------------------------
        # if True:
        #     return ItStage1
        # ------------------------------------------------------------------------------

        V01x, V02x, KV04x = self.feaExt(IE0.detach())
        Kt1x, Kt2x, KVt4x = self.feaExt(IEt.detach())
        V11x, V12x, KV14x = self.feaExt(IE1.detach())

        S04x, T04x, T02x, T01x = self.AttnMultiScale(Kt1x, Kt2x, KVt4x, V01x, V02x, KV04x,
                                                     self.subAttnMatC.detach())
        S02x = self.up2x(S04x)
        S01x = self.up4x(S04x)

        S14x, T14x, T12x, T11x = self.AttnMultiScale(Kt1x, Kt2x, KVt4x, V11x, V12x, KV14x,
                                                     self.subAttnMatC.detach())
        S12x = self.up2x(S14x)
        S11x = self.up4x(S14x)

        S4x = torch.where(S04x.ge(S14x), S04x, S14x)
        S2x = torch.where(S02x.ge(S12x), S02x, S12x)
        S1x = torch.where(S01x.ge(S11x), S01x, S11x)

        T4x = (S4x + 1) * torch.where(S04x.ge(S14x), T04x, T14x)  # 1, 128, H//4, W//4
        T2x = (S2x + 1) * torch.where(S02x.ge(S12x), T02x, T12x)  # 1, 64, H//2, W//2
        T1x = (S1x + 1) * torch.where(S01x.ge(S11x), T01x, T11x)  # 1, 32, H, W

        ItStage2 = self.AttDecoder(KVt4x, T4x, T2x, T1x, ST4x, ST2x, ST1x) + ItStage1

        return ItStage2

    def adap2Net(self, tensor: torch.Tensor):
        Height, Width = tensor.size(2), tensor.size(3)

        Height_ = int(math.floor(math.ceil(Height / self.netScale) * self.netScale))
        Width_ = int(math.floor(math.ceil(Width / self.netScale) * self.netScale))

        if any([Height_ != Height, Width_ != Width]):
            tensor = F.pad(tensor, [0, Width_ - Width, 0, Height_ - Height])

        return tensor


if __name__ == '__main__':
    from configs.configEVI import Config

    cfg = Config()
    dummyEventIt = torch.randn([1, inCh, 256, 256]).float().cuda()
    dummyEventI0 = torch.randn([1, inCh, 256, 256]).float().cuda()
    dummyEventI1 = torch.randn([1, inCh, 256, 256]).float().cuda()

    net = refineNet(cfg).train().cuda()

    output, T_lv1, T_lv2, T_lv3 = net(dummyEventI0, dummyEventI1, dummyEventIt)

    pass
