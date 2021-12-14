import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def backWarp(img: torch.Tensor, flow: torch.Tensor):
    device = img.device
    N, C, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.from_numpy(gridX).detach().to(device)
    gridY = torch.from_numpy(gridY).detach().to(device)

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


class forwadWarp(nn.Module):
    def __init__(self, bilinear=True):
        super(forwadWarp, self).__init__()

        self.bilinear = bilinear

    def forward(self, srcTensor: torch.Tensor, flow: torch.Tensor, weight: torch.Tensor = None):
        if weight is None:
            weight = torch.ones_like(srcTensor)

        srcTensor = srcTensor * weight

        self.device = srcTensor.device
        N, C, H, W = srcTensor.shape

        xx = torch.arange(0, W, requires_grad=False, device=self.device).view(1, 1, 1, -1).repeat(N, C, H, 1).float() \
             + flow[:, 0:1, :, :]
        yy = torch.arange(0, H, requires_grad=False, device=self.device).view(1, 1, -1, 1).repeat(N, C, 1, W).float() \
             + flow[:, 1:2, :, :]

        xxFloor = xx.floor().float().detach()
        xxCeil = xxFloor + 1.0

        yyFloor = yy.floor().float().detach()
        yyCeil = yyFloor + 1.0

        if self.bilinear:
            ltWeight, rtWeight, lbWeight, rbWeight = self.getBilinearWeight(xx, yy, xxFloor, yyFloor)
        else:
            ltWeight = torch.ones_like(srcTensor).detach()
            rtWeight = torch.ones_like(srcTensor).detach()
            lbWeight = torch.ones_like(srcTensor).detach()
            rbWeight = torch.ones_like(srcTensor).detach()

        ltImg = srcTensor * ltWeight
        rtImg = srcTensor * rtWeight
        lbImg = srcTensor * lbWeight
        rbImg = srcTensor * rbWeight

        ltNorm = weight * ltWeight
        rtNorm = weight * rtWeight
        lbNorm = weight * lbWeight
        rbNorm = weight * rbWeight

        ltTarget, ltScaler = self.splatting(xxFloor, yyFloor, ltImg, ltNorm)
        rtTarget, rtScaler = self.splatting(xxCeil, yyFloor, rtImg, rtNorm)
        lbTarget, lbScaler = self.splatting(xxFloor, yyCeil, lbImg, lbNorm)
        rbTarget, rbScaler = self.splatting(xxCeil, yyCeil, rbImg, rbNorm)

        scale = ltScaler + rtScaler + lbScaler + rbScaler
        remapTensor = torch.zeros_like(srcTensor)

        nonZero = scale != 0
        remapTensor[nonZero] = (ltTarget[nonZero] + rtTarget[nonZero] + lbTarget[nonZero] + rbTarget[nonZero]) / scale[
            nonZero]
        # remapTensor = ltTarget + rtTarget + lbTarget + rbTarget

        # eps = 1e-8
        # remapTensor = (ltTarget + rtTarget + lbTarget + rbTarget + eps) / (scale + eps)

        # return remapTensor, scale
        return remapTensor

    def getBilinearWeight(self, xx, yy, xxFloor, yyFloor):
        alpha = xx - xxFloor
        beta = yy - yyFloor
        ltWeight = (1 - alpha) * (1 - beta)
        rtWeight = alpha * (1 - beta)
        lbWeight = (1 - alpha) * beta
        rbWeight = alpha * beta
        return ltWeight, rtWeight, lbWeight, rbWeight

    def splatting(self, xx, yy, img, Norm):
        N, C, H, W = xx.shape

        nn = torch.arange(0, N, requires_grad=False, device=self.device).view(N, 1, 1, 1).long(). \
            repeat(1, C, H, W)  # NCHW
        cc = torch.arange(0, C, requires_grad=False, device=self.device).view(1, C, 1, 1).long(). \
            repeat(N, 1, H, W)  # NCHW

        # grid = xx + yy * W
        stride = xx.stride()

        # grid = nn * C * H * W + cc * H * W + yy * W + xx
        grid = nn * stride[0] + cc * stride[1] + yy.long() * stride[2] + xx.long()

        mask = (xx.ge(0) & xx.lt(W)) & (yy.ge(0) & yy.lt(H))

        gridSelect = grid.masked_select(mask).long()

        targetImg = torch.zeros_like(img).float()

        scaler = torch.zeros_like(Norm).float()

        targetImg.put_(gridSelect, img.masked_select(mask), accumulate=True)
        scaler.put_(gridSelect, Norm.masked_select(mask), accumulate=True)

        return targetImg, scaler
    
# if __name__ == '__main__':
#     img = cv2.imread('./135_left.jpg')
#     flow = utils.readFlowFile('135.flo')
#     devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     Func = ForwadWarpLayer()
#     H, W, C = img.shape
#
#     imgTensor = torch.from_numpy(img).float().to(devices).unsqueeze(0).contiguous()
#     flowTensor = torch.from_numpy(flow).float().to(devices).unsqueeze(0).contiguous()
#
#     # imgTensor = imgTensor.permute([0, 3, 1, 2])
#     # flowTensor = flowTensor.permute([0, 3, 1, 2])
#
#     remapTensor = Func(imgTensor, flowTensor)
#
#     remapImage = (remapTensor[0, ...].to('cpu').numpy()).astype(np.uint8)
#     cv2.namedWindow('1', 0)
#     cv2.imshow('1', remapImage)
#     cv2.waitKey(0)
