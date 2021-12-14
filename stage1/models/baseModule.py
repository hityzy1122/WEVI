import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn import init
from collections import OrderedDict
import numpy as np
import correlation_cuda
from torch.nn import Parameter
from torch.optim import lr_scheduler
from lib.warp import backWarp
from lib.visualTool import viz


# for Base option -------------------------------------------------------------------------------

class Interp(nn.Module):
    def __init__(self, scale=None, size=None):
        super(Interp, self).__init__()
        self.scale = scale
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale, mode='bilinear', align_corners=True)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def getPad(in_, kernel_size, stride, dilation=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + dilation * (kernel_size - 1) + 1 - in_) / 2)


def getNorm(name: str, num_features: int = 0, affine=True):
    if name.lower() == 'instance':
        return nn.InstanceNorm2d(num_features=num_features, affine=affine)
    if name.lower() == 'lrn':
        return nn.LocalResponseNorm(8)
    if name.lower() == 'bn':
        return nn.BatchNorm2d(num_features=num_features, affine=affine)
    if name.lower() == 'identity':
        return nn.Identity()
    if name.lower() == 'group':
        return torch.nn.GroupNorm(num_groups=4, num_channels=num_features, affine=affine)


def getAct(name: str, num_parameters: int = 0, inplace=True):
    if name.lower() == 'relu':
        return nn.ReLU(inplace=inplace)
    if name.lower() == 'prelu':
        return nn.PReLU(num_parameters)
    if name.lower() == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if name.lower() == 'swish':
        return Swish()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation: tuple = (1, 1), groups=1, padModel='replicate', cfg=None):
        super(ConvBlock, self).__init__()

        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size
        else:
            kernel_size = (kernel_size, kernel_size)
        self.bias = False if 'identity' == cfg.netNorm else True
        self.padModel = padModel

        self.padList = [(kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2,
                        (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2]

        self.conv = nn.Sequential(nn.ReplicationPad2d(self.padList),
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, dilation=dilation, groups=groups, bias=self.bias)
                                  )
        self.norm2d = getNorm(cfg.netNorm, num_features=out_channels)
        self.activate = getAct(cfg.netActivate, num_parameters=out_channels)

    def forward(self, x):

        # x = F.pad(input=x, pad=self.padList, mode=self.padModel)
        x = self.conv(x)
        x = self.norm2d(x)
        x = self.activate(x)

        return x


# end BaseOption -----------------------------------------------------------------------------------


# for pwcNet ---------------------------------------------------------------------------------------
class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, *agrs):
        input1 = agrs[0]
        input2 = agrs[1]

        ctx.pad_size = agrs[2]

        ctx.kernel_size = agrs[3]
        ctx.max_displacement = agrs[4]
        ctx.stride1 = agrs[5]
        ctx.stride2 = agrs[6]
        ctx.corr_multiply = agrs[7]

        ctx.save_for_backward(input1, input2)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(
                input1,
                input2,
                rbot1,
                rbot2,
                output,
                ctx.pad_size,
                ctx.kernel_size,
                ctx.max_displacement,
                ctx.stride1,
                ctx.stride2,
                ctx.corr_multiply)

        return output

    @staticmethod
    def backward(ctx, *grad_output):
        grad_output = grad_output[0]
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(
                input1,
                input2,
                rbot1,
                rbot2,
                grad_output,
                grad_input1,
                grad_input2,
                ctx.pad_size,
                ctx.kernel_size,
                ctx.max_displacement,
                ctx.stride1,
                ctx.stride2,
                ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(Module):
    def __init__(
            self,
            pad_size=0,
            kernel_size=0,
            max_displacement=0,
            stride1=1,
            stride2=2,
            corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        Correlation = CorrelationFunction.apply
        result = Correlation(
            input1,
            input2,
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.corr_multiply)

        return result


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1, inplace=True))


# end pwcNet --------------------------------------------------------------------------------------------------

# for fuse Net----------------------------------------------------------------------------------------------
class BaseNet(nn.Module):
    """
    init pretrained weights with weight file
    init other weights with a certain random pattern
    """

    def __init__(self, netInitType: str = 'xavier', netInitGain=0.2, netNormAffine=True):
        super(BaseNet, self).__init__()
        self.netInitType = netInitType
        self.netInitGain = netInitGain
        self.netNormAffine = netNormAffine

    def forward(self, *input):
        raise NotImplementedError

    def randomInitNet(self):
        # init all weights by pre-defined pattern firstly
        for m in self.modules():
            if any([isinstance(m, nn.Conv2d), isinstance(m, nn.ConvTranspose2d), isinstance(m, nn.Linear)]):
                if self.netInitType == 'normal':
                    init.normal_(m.weight, 0.0, self.netInitGain)
                elif self.netInitType == 'xavier':
                    init.xavier_normal_(m.weight, gain=self.netInitGain)
                elif self.netInitType == 'kaiming':
                    init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                elif self.netInitType == 'orthogonal':
                    init.orthogonal_(m.weight, gain=self.netInitGain)
                elif self.netInitType == 'default':
                    pass
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif any([isinstance(m, nn.InstanceNorm2d), isinstance(m, nn.LocalResponseNorm),
                      isinstance(m, nn.BatchNorm2d), isinstance(m, nn.GroupNorm)]) and self.netNormAffine:
                try:
                    init.constant_(m.weight, 1.0)
                    init.constant_(m.bias, 0.0)
                except Exception as e:
                    pass

    def initPreweight(self, pathPreWeight: str = None, rmModule=True):

        preW = self.getWeight(pathPreWeight)
        assert preW is not None, 'weighth in {} is empty'.format(pathPreWeight)
        modelW = self.state_dict()
        preWDict = OrderedDict()
        # modelWDict = OrderedDict()

        for k, v in preW.items():
            if rmModule:
                preWDict[k.replace('module.', "")] = v
            else:
                preWDict[k] = v

        shareW = {k: v for k, v in preWDict.items() if str(k) in modelW}
        assert shareW, 'shareW is empty'
        self.load_state_dict(preWDict, strict=False)

    @staticmethod
    def getWeight(pathPreWeight: str = None):
        if pathPreWeight is not None:
            return torch.load(pathPreWeight, map_location=torch.device('cpu'))
        else:
            return None

    @staticmethod
    def padToScale(img, netScale):
        _, _, h, w = img.size()
        oh = int(np.ceil(h * 1.0 / netScale) * netScale)
        ow = int(np.ceil(w * 1.0 / netScale) * netScale)
        img = F.pad(img, [0, ow - w, 0, oh - h], mode='reflect')
        return img


class conv3x3(nn.Module):
    def __init__(self, in_c, out_c, cfg, ks=2):
        super(conv3x3, self).__init__()

        self.conv = nn.Sequential(
            nn.ReplicationPad2d([1, 1, 1, 1]),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=ks, bias=False)
        )
        self.norm2d = getNorm(cfg.netNorm, num_features=out_c)
        self.activate = getAct(cfg.netActivate, num_parameters=out_c)

    def forward(self, x):
        # x = F.pad(input=x, pad=[2, 2, 2, 2], mode='replicate')
        x = self.conv(x)
        x = self.norm2d(x)
        x = self.activate(x)

        return x


class deconv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, cfg=None):
        super(deconv, self).__init__()

        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size
        else:
            kernel_size = (kernel_size, kernel_size)
        self.bias = False if 'identity' == cfg.netNorm else True

        self.padList = [(kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2,
                        (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2]

        self.deconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReplicationPad2d(self.padList),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=1, bias=False)
        )

        self.norm2d = getNorm(cfg.netNorm, num_features=out_c)
        self.activate = getAct(cfg.netActivate, num_parameters=out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):

        x = self.deconv(x)
        x = self.norm2d(x)
        x = self.activate(x)

        return torch.cat([x, skip], dim=1)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, cfg=None):
        super(DeconvBlock, self).__init__()

        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size
        else:
            kernel_size = (kernel_size, kernel_size)
        self.bias = False if 'identity' == cfg.netNorm else True

        self.padList = [(kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2,
                        (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2]

        self.deconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReplicationPad2d(self.padList),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=1, bias=False)
        )

        self.norm2d = getNorm(cfg.netNorm, num_features=out_c)
        self.activate = getAct(cfg.netActivate, num_parameters=out_c)

    def forward(self, x: torch.Tensor):

        x = self.deconv(x)
        x = self.norm2d(x)
        x = self.activate(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, cfg):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(in_c=num_channels, out_c=num_channels, kernel_size=kernel_size, cfg=cfg)
        self.up_conv2 = ConvBlock(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size,
                                  stride=2, cfg=cfg)
        self.up_conv3 = DeconvBlock(in_c=num_channels, out_c=num_channels, kernel_size=kernel_size, cfg=cfg)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, cfg):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(in_channels=num_channels, out_channels=num_channels,
                                    kernel_size=kernel_size, stride=2, cfg=cfg)
        self.down_conv2 = DeconvBlock(in_c=num_channels, out_c=num_channels, kernel_size=kernel_size, cfg=cfg)
        self.down_conv3 = ConvBlock(in_channels=num_channels, out_channels=num_channels,
                                    kernel_size=kernel_size, stride=2, cfg=cfg)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class FuseLayer(nn.Module):
    def __init__(self, c_h, c_ef):
        super(FuseLayer, self).__init__()
        self.c_ef = c_ef
        self.c_h = c_h

        self.norm = nn.InstanceNorm2d(c_h, affine=False)

        self.convE1 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)
        self.convE2 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)

        self.convF1 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)
        self.convF2 = nn.Conv2d(c_ef, c_h, kernel_size=1, stride=1, padding=0, bias=True)

        self.convMask = nn.Sequential(nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
                                      nn.Conv2d(c_h, 1, kernel_size=3, stride=1, bias=True),
                                      nn.Sigmoid())

    def forward(self, h_in, z_e, z_f):
        h = self.norm(h_in)

        gammaE = self.convE1(z_e)
        betaE = self.convE2(z_e)
        E = gammaE * h + betaE

        gammaF = self.convF1(z_f)
        betaF = self.convF2(z_f)
        F = gammaF * h + betaF

        hEF = h
        M = self.convMask(hEF)
        out = M * E + (1.0 - M) * F

        return out


class FuseBlock(nn.Module):
    def __init__(self, cin, cout, c_ef):
        super(FuseBlock, self).__init__()
        self.cin = cin
        self.cout = cout

        self.AAD1 = FuseLayer(cin, c_ef)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, bias=False)
        )

        self.AAD2 = FuseLayer(cin, c_ef)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, bias=False)
        )

        if cin != cout:
            self.AAD3 = FuseLayer(cin, c_ef)
            self.conv3 = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.ReplicationPad2d(padding=[1, 1, 1, 1]),
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, bias=False)
            )

    def forward(self, h, z_e, z_f):

        x = self.AAD1(h, z_e, z_f)
        x = self.conv1(x)

        x = self.AAD2(x, z_e, z_f)
        x = self.conv2(x)

        if self.cin != self.cout:
            h = self.AAD3(h, z_e, z_f)
            h = self.conv3(h)
        x = x + h

        return x


# end fuseNet --------------------------------------------------------------------------------------

# for PostProcess ---------------------------------------------------------------------------------


# end PostProcess ---------------------------------------------------------------------------------

if __name__ == '__main__':
    from configs import configEVI

    cfg = configEVI.Config()
    pass
