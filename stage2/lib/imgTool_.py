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
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def crop(img, x, y, h, w):
    """Crop the given CV Image.

    Args:
        img (np.ndarray): Image to be cropped.
        x: Upper pixel coordinate.
        y: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image. Got {}'.format(type(img))
    assert h > 0 and w > 0, 'h={} and w={} should greater than 0'.format(h, w)

    x1, y1, x2, y2 = round(x), round(y), round(x + h), round(y + w)

    try:
        check_point1 = img[x1, y1, ...]
        check_point2 = img[x2 - 1, y2 - 1, ...]
    except IndexError:
        # warnings.warn('crop region is {} but image size is {}'.format((x1, y1, x2, y2), img.shape))
        img = cv2.copyMakeBorder(img, - min(0, x1), max(x2 - img.shape[0], 0),
                                 -min(0, y1), max(y2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)

    finally:
        return img[x1:x2, y1:y2, ...].copy()


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
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def method(self, pic: np.ndarray):
        if len(pic.shape) == 2:
            pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(pic.copy()).permute(2, 0, 1).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float().div(255.0)
        else:
            return img.float()

    def __call__(self, pic: np.ndarray):

        if not isinstance(pic, list):
            return self.method(pic)
        else:
            # [0~1]
            return [self.method(i) for i in pic]

    def __repr__(self):
        return self.__class__.__name__ + '()'


def ToCVImage(tensor: torch.Tensor):
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor[0, ...].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    tensor = np.ascontiguousarray(tensor)
    return tensor


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def method(self, tensor):

        if _is_tensor_image(tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor
        elif _is_numpy_image(tensor):
            return (tensor.astype(np.float32) - 255.0 * np.array(self.mean)) / np.array(self.std)
        else:
            raise RuntimeError('Undefined type')

    def __call__(self, tensor):
        realimg = tensor[0:-4]
        dct = tensor[-4::]
        return [self.method(i) for i in realimg] + dct

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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

    def __init__(self, size, interpolation='BILINEAR'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))
        if not (isinstance(self.size, int) or (isinstance(self.size, collections.Iterable) and len(self.size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))

        if isinstance(self.size, int):
            h, w, c = img.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[self.interpolation])
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[self.interpolation])
        else:
            oh, ow = self.size
            return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[self.interpolation])

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
    """Crop the given CV Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (CV Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        try:
            i = random.randint(0, h - th)
        except ValueError:
            i = random.randint(h - th, 0)
        try:
            j = random.randint(0, w - tw)
        except ValueError:
            j = random.randint(w - tw, 0)
        return i, j, th, tw

    def method(self, img, i, j, h, w):
        """
        Args:
            img (np.ndarray): Image to be cropped.

        Returns:
            np.ndarray: Cropped image.
        """
        if self.padding > 0:
            img = Pad(self.padding)(img)

            # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = Pad((int((1 + self.size[1] - img.shape[1]) / 2), 0))(img)
            # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = Pad((0, int((1 + self.size[0] - img.shape[0]) / 2)))(img)

        return crop(img, i, j, h, w)

    def __call__(self, img):

        i, j, h, w = self.get_params(img[0], self.size)

        if not isinstance(img, list):
            return self.method(img, i, j, h, w)
        else:
            return [self.method(im, i, j, h, w) for im in img]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def method(self, img):
        """Horizontally flip the given PIL Image.

            Args:
                img (np.ndarray): Image to be flipped.

            Returns:
                np.ndarray:  Horizontall flipped image.
            """
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        return np.flip(img, axis=1)

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            # U channel of dct should be inversed
            img[-4] = -img[-4]
            img[-2] = -img[-2]

            if not isinstance(img, list):
                return self.method(img)
            else:
                return [self.method(i) for i in img]
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
        realimg = img[0:-4]
        dct = img[-4::]
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        return transform(realimg) + dct


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
        realimg = img[0:-4]
        dct = img[-4::]
        img_shape = img[0].shape
        self.get_params(img_shape)
        return [self.method(i) for i in realimg] + dct

    def __repr__(self):
        return self.__class__.__name__
