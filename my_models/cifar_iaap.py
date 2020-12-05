"""
CNN model for 32x32x3 image classification
"""
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from art.classifiers import PyTorchClassifier

from armory.data.utils import maybe_download_weights_from_s3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocessing_fn(img):
    # Model will trained with inputs normalized from 0 to 1
    img = img.astype(np.float32) / 255.0
    return img

def _is_tensor_image(img):
    return torch.is_tensor(img)

def data_normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
       This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
       tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
       mean (sequence): Sequence of means for each channel.
       std (sequence): Sequence of standard deviations for each channel.
       inplace(bool,optional): Bool to make this operation inplace.
    Returns:
       Tensor: Normalized Tensor image.
    """

    if not _is_tensor_image(tensor):
       raise TypeError('tensor is not a torch image.')

    if not inplace:
       tensor = tensor.clone()

    # print("Shape of the tensor is: ", tensor.size())
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

def init_weight(*args):
   return nn.Parameter(nn.init.kaiming_normal_(torch.zeros(*args), mode='fan_out', nonlinearity='relu'))


class Block(nn.Module):
    """
    Pre-activated ResNet block.
    """
    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv0', init_weight(width, width, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self.conv0, padding=1)
        h = F.conv2d(F.relu(self.bn1(h)), self.conv1, padding=1)
        return x + h


class DownsampleBlock(nn.Module):
    """
    Downsample block.
    Does F.avg_pool2d + torch.cat instead of strided conv.
    """

    def __init__(self, width):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(width // 2, affine=False)
        self.register_parameter('conv0', init_weight(width, width // 2, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self.conv0, padding=1, stride=2)
        h = F.conv2d(F.relu(self.bn1(h)), self.conv1, padding=1)
        x_d = F.avg_pool2d(x, kernel_size=3, padding=1, stride=2)
        x_d = torch.cat([x_d, torch.zeros_like(x_d)], dim=1)
        return x_d + h


class WRN(nn.Module):
    """
    Implementation of modified Wide Residual Network.
    Differences with pre-activated ResNet and Wide ResNet:
       * BatchNorm has no affine weight and bias parameters
       * First layer has 16 * width channels
       * Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
       * Downsample is done by F.avg_pool2d + torch.cat instead of strided conv
    First and last convolutional layers are kept in float32.
    """

    def __init__(self, depth=32, width=10, num_classes=10):
        super().__init__()
        widths = [int(v * width) for v in (16, 32, 64)]
        n = (depth - 2) // 6
        self.register_parameter('conv0', init_weight(widths[0], 3, 3, 3))
        self.group0 = self._make_block(widths[0], n)
        self.group1 = self._make_block(widths[1], n, downsample=True)
        self.group2 = self._make_block(widths[2], n, downsample=True)
        self.bn = nn.BatchNorm2d(widths[2], affine=False)
        self.register_parameter('conv_last', init_weight(num_classes, widths[2], 1, 1))
        self.bn_last = nn.BatchNorm2d(num_classes)
        self.mean = [125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0]
        self.std = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]


    def _make_block(self, width, n, downsample=False):
        def select_block(j):
            if downsample and j == 0:
                return DownsampleBlock(width)
            return Block(width)

        return nn.Sequential(OrderedDict(('block%d' % i, select_block(i)) for i in range(n)))


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = data_normalize(x, self.mean, self.std)
        h = F.conv2d(x, self.conv0, padding=1)
        h = self.group0(h)
        h = self.group1(h)
        h = self.group2(h)
        h = F.relu(self.bn(h))
        h = F.conv2d(h, self.conv_last)
        h = self.bn_last(h)
        output = F.avg_pool2d(h, kernel_size=h.shape[-2:]).view(h.shape[0], -1)
        # _, pred = torch.max(output, dim=1)
        return output

def make_cifar_model(**kwargs):

    model = WRN()
    model.eval()
    return model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_cifar_model(**model_kwargs)
    print("Pytorch version is: {}".format(torch.__version__))
    # exit(-1)
    model.to(DEVICE)
    print("DEVICES is: {}".format(DEVICE))
    # exit(-1)

    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        checkpoint = torch.load(filepath, map_location=DEVICE)
        new_model = {}
        new_model['model'] = {}
        for key in checkpoint['model'].keys():
            new_model['model'][key[7:]] = checkpoint['model'][key]

        model.load_state_dict(new_model['model'])

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=checkpoint['optimizer'],
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model