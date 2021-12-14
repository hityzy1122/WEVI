import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F
from pathlib import Path
from torch.autograd import Function
import correlation_cuda
from torch.nn import init
from torch.nn.modules.module import Module
from collections import OrderedDict


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

        # for k, v in modelW.items():
        #     modelWDict[k.replace('module.', "")] = v

        # shareW = {k: v for k, v in preWDict.items() if str(k) in modelWDict}
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


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class PWCDCNet(BaseNet):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, md=4, cfg=None):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet, self).__init__()
        self.mean = cfg.trainMean if cfg is not None else 0
        self.std = cfg.trainStd if cfg is not None else 1
        self.netScale = 64.0

        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)

        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 128 + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 96 + 4
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 64 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 32 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        self.randomInitNet()

        pathPreWeight = str(Path(__file__).parent.absolute() / Path('pwc_net.pth.tar'))
        self.initPreweight(pathPreWeight)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def getWeight(self, pathPreWeight: str = None):

        data = torch.load(pathPreWeight, map_location=torch.device('cpu'))
        if 'state_dict' in data.keys():
            return data['state_dict']
        else:
            return data

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1] 
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, 'bilinear')
        mask = torch.ones(x.size()).cuda()
        mask = F.grid_sample(mask, vgrid, 'bilinear')

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def adap2pwcNet(self, tensor: torch.Tensor):

        out = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)

        return out

    def forward(self, tFirst, tSecond, iters=20, test_mode=True):

        tFirst = self.adap2pwcNet(tFirst)
        tSecond = self.adap2pwcNet(tSecond)

        Width, Height = tFirst.size(3), tFirst.size(2)

        Width_ = int(2 * math.floor(math.ceil(Width / self.netScale) * self.netScale))
        Height_ = int(2 * math.floor(math.ceil(Height / self.netScale) * self.netScale))

        tFirst_ = F.interpolate(input=tFirst, size=(Height_, Width_), mode='bilinear', align_corners=True)
        tSecond_ = F.interpolate(input=tSecond, size=(Height_, Width_), mode='bilinear', align_corners=True)

        c11 = self.conv1b(self.conv1aa(self.conv1a(tFirst_)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(tSecond_)))

        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))

        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))

        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))

        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))

        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)

        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        # add-------------------------------------------------------------------

        flow2Up = 20.0 * F.interpolate(input=flow2, size=(Height, Width), mode='bilinear', align_corners=True)

        flow2Up[:, 0, :, :] *= float(Width) / float(Width_)
        flow2Up[:, 1, :, :] *= float(Height) / float(Height_)

        # flow2Up = 20.0 * F.interpolate(input=flow2, scale_factor=4, mode='bilinear', align_corners=True)
        # flow2Up = flow2Up[:, :, 0:Height, 0:Width]
        return flow2Up


if __name__ == '__main__':
    a = torch.randn([1, 3, 128, 128]).cuda()
    b = torch.randn([1, 3, 128, 128]).cuda()
    net = PWCDCNet(cfg=None).cuda()
    c = net(a, b)
    pass