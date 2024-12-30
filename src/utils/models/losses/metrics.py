import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`."""
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`."""
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape.')

    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [
            convolve(im, downsample_filter, mode='reflect')
            for im in [im1, im2]
        ]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1]**weights[0:levels - 1]) *
            (mssim[levels - 1]**weights[levels - 1]))

from collections import OrderedDict
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models


def normalize_activation(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize activations to unit length."""
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1') -> OrderedDict:
    """Download and prepare the state dictionary for the specified network."""
    url = (
        f"https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        f"master/lpips/weights/v{version}/{net_type}.pth"
    )
    state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    return OrderedDict((k.replace("lin", "").replace("model.", ""), v) for k, v in state_dict.items())


class BaseNet(nn.Module):
    """Base network class for LPIPS."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("std", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)
        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class AlexNet(BaseNet):
    """AlexNet for LPIPS."""
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]
        self.set_requires_grad(False)


class SqueezeNet(BaseNet):
    """SqueezeNet for LPIPS."""
    def __init__(self):
        super().__init__()
        self.layers = models.squeezenet1_1(pretrained=True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]
        self.set_requires_grad(False)


class VGG16(BaseNet):
    """VGG16 for LPIPS."""
    def __init__(self):
        super().__init__()
        self.layers = models.vgg16(pretrained=True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]
        self.set_requires_grad(False)


class LinLayers(nn.ModuleList):
    """Linear layers for computing LPIPS."""
    def __init__(self, n_channels_list: Sequence[int]):
        super().__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])
        for param in self.parameters():
            param.requires_grad = False


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) criterion."""
    def __init__(self, net_type: str = "alex", version: str = "0.1"):
        super().__init__()
        assert version == "0.1", "Only version 0.1 is supported."
        self.net = {"alex": AlexNet, "squeeze": SqueezeNet, "vgg": VGG16}[net_type]()
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feat_x, feat_y = self.net(x), self.net(y)
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        return torch.sum(torch.cat(res, 0), 0, True)


x = torch.rand(1, 3, 256, 256)  # Example input tensor
y = torch.rand(1, 3, 256, 256)  # Example input tensor

lpips_criterion = LPIPS(net_type="alex")
loss = lpips_criterion(x, y)
print(f"LPIPS Loss: {loss.item()}")
