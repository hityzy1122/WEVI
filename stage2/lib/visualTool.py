import numpy as np
import cv2
import torch
import torch.nn.functional as F


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def viz(img: torch.Tensor = None, name='1', wait=0):
    img = img.cpu().detach().float()[0].permute([1,2,0])
    img = (255*(img - img.min()) / (img.max() - img.min())).byte().numpy()
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)
    cv2.waitKey(wait)


def flow2ImgBatch(flos: torch.Tensor):
    N, C, H, W = flos.shape
    output = np.zeros([N, H, W, 3])
    for n in range(N):
        flo = flos[n, ...].permute(1, 2, 0).cpu().numpy()
        output[n, ...] = flow_to_image(flo)
    return output


def mask2ImgBatch(mask: torch.Tensor):
    N, C, H, W = mask.shape
    output = np.zeros([N, H, W, 3])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask * 255).cpu().byte().numpy()
    for n in range(N):
        output[n, ...] = cv2.applyColorMap(mask[n, 0], cv2.COLORMAP_JET)
    return output.astype(np.uint8)


def bis(input, dim, index):
    N, C, H, W = input.shape
    input = F.unfold(input, kernel_size=(24, 24), padding=8, stride=8)
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
    expanse = list(input.size())
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    input = torch.gather(input, dim, index)

    input = F.fold(input, output_size=(H, W), kernel_size=(24, 24), padding=8, stride=8)

    return input


def softAttn(input, R4x):
    N, C, H, W = input.shape
    n = 1
    # R4x = torch.softmax(R4x / 0.01, dim=1)

    input_ubfold = F.unfold(input, kernel_size=(24, 24), padding=8, stride=8)
    input_ubfold = input_ubfold.view([1, 1728, 54, 96])
    R4x = R4x.view(1, 54, 96, 54, 96)
    output = torch.zeros([1, 1728, 54, 96])

    for x in range(n, output.shape[2] - n - 1):
        for y in range(n, output.shape[3] - n - 1):
            subWeight = R4x[:, x - n:x + n + 1, y - n:y + n + 1, x, y].unsqueeze(1)
            # subWeight = F.softmax(subWeight.contiguous().view([1, 1, -1])/0.00001, dim=-1).view([1, 1, 2*n+1, 2*n+1])
            subWeight = F.softmax(subWeight.contiguous().view([1, 1, -1])/0.15, dim=-1).view(
                [1, 1, 2 * n + 1, 2 * n + 1])
            subPatch = input_ubfold[:, :, x - n:x + n + 1, y - n:y + n + 1]
            patch = subWeight * subPatch

            weight = subWeight.sum()

            patch = patch.sum(2).sum(2) / weight
            output[:, :, x, y] = patch

    output = F.fold(output.view([1, 1728, -1]), output_size=(H, W), kernel_size=(24, 24), padding=8, stride=8)
    outPut = output.view([N, C, H, W])
    viz(outPut)
    return outPut