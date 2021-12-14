from __future__ import division
import torch
import math
import random
import cv2
import numpy as np
import numbers
import collections
import imageio

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }


def imglist2GIF(imglist: list, gifPath: str):
    imageio.mimsave(gifPath, imglist, 'GIF', duration=0.5)


def makeGrid(imgList: list, shape=(3, 3)):
    assert len(imgList) == shape[0] * shape[1], \
        'number of img is {}, grid shape is {}'.format(len(imgList), shape[0] * shape[1])
    img = np.stack(imgList, axis=0)
    N, H, W, C = img.shape
    nh = shape[0]
    nw = shape[1]
    img = img.reshape((nh, nw, H, W, C)).swapaxes(1, 2).reshape(nh * H, nw * W, C)
    return img


def flowToImg(flow: torch.Tensor, normalize=True, info=None, flow_mag_max=None, idx=0):
    flow = flow[0, ...].permute(1, 2, 0).cpu().detach().numpy()
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(
        flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
            # hsv[..., 2] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
        # hsv[..., 2] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('flow{}'.format(idx), 0)
    cv2.imshow('flow{}'.format(idx), img)
    cv2.waitKey(0)


def _is_tensor_image(img):
    # return torch.is_tensor(img) and img.ndimension() == 3
    return torch.is_tensor(img)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, pic: np.ndarray):
        img: torch.Tensor = torch.from_numpy(pic.copy()).permute(0, 3, 1, 2).contiguous()

        return img.float().div(255.0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def toCVImage(tensor: torch.Tensor):
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor[0, ...].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    tensor = np.ascontiguousarray(tensor)
    return tensor


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float().view(1, -1, 1, 1)
        self.std = torch.tensor(std).float().view(1, -1, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor.float() - self.mean) / self.std
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class inNormalize(object):
    def __init__(self, mean: list, std: list):
        # self.mean = mean
        # self.std = std
        self.meanTensor = torch.tensor(mean).float().view(1, -1, 1, 1)
        self.stdTensor = torch.tensor(std).float().view(1, -1, 1, 1)

    def __call__(self, tensor: torch.Tensor):
        device0 = tensor.device
        meanTensor = self.meanTensor.to(device0).detach()
        stdTensor = self.stdTensor.to(device0).detach()

        output = tensor * stdTensor + meanTensor

        return output


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``
    """

    def __init__(self, factor, interpolation='BILINEAR'):
        assert isinstance(factor, int)
        self.factor = factor
        self.interpolation = interpolation

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))
        if not (isinstance(self.factor, int) or (
                isinstance(self.factor, collections.Iterable) and len(self.factor) == 2)):
            raise TypeError('Got inappropriate factor arg: {}'.format(self.factor))
        h, w, c = img.shape
        if ((h // self.factor * self.factor) != h) or ((w // self.factor * self.factor) != w):
            oh = math.ceil(h / 32.0) * 32
            ow = math.ceil(w / 32.0) * 32

            return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[self.interpolation])
        else:
            return img

    def __call__(self, img):
        if not isinstance(img, list):
            return self.method(img)
        else:
            return [self.method(i) for i in img]

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Pad(object):
    """Pad the given CV Image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value at the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def method(self, img):

        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        if not isinstance(self.padding, (numbers.Number, tuple)):
            raise TypeError('Got inappropriate padding arg')
        if not isinstance(self.fill, (numbers.Number, str, tuple)):
            raise TypeError('Got inappropriate fill arg')
        if not isinstance(self.padding_mode, str):
            raise TypeError('Got inappropriate padding_mode arg')

        if isinstance(self.padding, collections.Sequence) and len(self.padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(self.padding)))

        assert self.padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
            'Padding mode should be either constant, edge, reflect or symmetric'

        if isinstance(self.padding, int):
            pad_left = pad_right = pad_top = pad_bottom = self.padding
        if isinstance(self.padding, collections.Sequence) and len(self.padding) == 2:
            pad_left = pad_right = self.padding[0]
            pad_top = pad_bottom = self.padding[1]
        if isinstance(self.padding, collections.Sequence) and len(self.padding) == 4:
            pad_left, pad_top, pad_right, pad_bottom = self.padding

        if isinstance(self.fill, numbers.Number):
            self.fill = (self.fill,) * (2 * len(img.shape) - 3)

        if self.padding_mode == 'constant':
            assert (len(self.fill) == 3 and len(img.shape) == 3) or (len(self.fill) == 1 and len(img.shape) == 2), \
                'channel of image is {} but length of fill is {}'.format(img.shape[-1], len(self.fill))

        img = cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                 borderType=PAD_MOD[self.padding_mode], value=self.fill)
        return img

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be padded.

        Returns:
            CV Image: Padded image.
        """
        if not isinstance(img, list):
            return self.method(img)
        else:
            return [self.method(i) for i in img]

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)


class RandomTransforms(object):
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.2):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomCrop(object):
    __slots__ = ['size']

    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, img):
        n, h, w, _ = img.shape
        th, tw = self.size

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return img[:, i:i + th, j:j + tw, :]


class RandomHorizontalFlip(object):
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    __slots__ = ['p']

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray):
        # N, H, W, C = img.shape
        if random.random() < self.p:
            img = np.flip(img, axis=2)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class AdjustBrightness(object):
    __slots__ = ['brightness_factor']

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        im = img.astype(np.float32) * self.brightness_factor
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustContrast(object):
    __slots__ = ['contrast_factor']

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))
        im = img.astype(np.float32)
        mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
        im = (1 - self.contrast_factor) * mean + self.contrast_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustSaturation(object):
    __slots__ = ['saturation_factor']

    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        im = img.astype(np.float32)
        degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        im = (1 - self.saturation_factor) * degenerate + self.saturation_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustHue(object):
    __slots__ = ['hue_factor']

    def __init__(self, hue_factor):
        self.hue_factor = hue_factor

    def method(self, img):
        if not (-0.5 <= self.hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(self.hue_factor))

        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        im = img.astype(np.uint8)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
        hsv[..., 0] += np.uint8(self.hue_factor * 255)

        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return im.astype(img.dtype)

    def __call__(self, img):

        return [self.method(i) for i in img]


class RandomColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    __slots__ = ['brightness', 'contrast', 'saturation', 'hue']

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(RandomApply([AdjustBrightness(brightness_factor)]))
        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(RandomApply([AdjustContrast(contrast_factor)]))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(RandomApply([AdjustSaturation(saturation_factor)]))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(RandomApply([AdjustHue(hue_factor)]))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Color jittered image.

        """

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        return transform(img)


class GaussianNoise(object):
    __slots__ = ['means', 'stds', 'gauss']

    def __init__(self, mean=0, std=0.05):
        assert isinstance(mean, numbers.Number) and mean >= 0, 'mean should be a positive value'
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        self.means = mean
        self.stds = std

    def get_params(self, img_shape):
        mean = random.uniform(-self.means, self.means)
        std = random.uniform(0, self.stds)
        self.gauss = np.random.normal(mean, std, img_shape).astype(np.float32)

    def method(self, img):
        if len(img.shape) == 3:
            imgtype = img.dtype
            noisy = np.clip((1 + self.gauss) * img.astype(np.float32), 0, 255)
            return noisy.astype(imgtype)
        else:
            return img

    def __call__(self, img):

        img_shape = img[0].shape
        self.get_params(img_shape)
        return [self.method(i) for i in img]

    def __repr__(self):
        return self.__class__.__name__
