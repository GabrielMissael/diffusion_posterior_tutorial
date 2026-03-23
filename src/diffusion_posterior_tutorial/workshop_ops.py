from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SOURCE_NPIX = 64
OBS_NPIX = 64


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_batch_images(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 2:
        return images.unsqueeze(0)
    if images.ndim == 3:
        return images
    if images.ndim == 4 and images.shape[1] == 1:
        return images[:, 0]
    raise ValueError(f"Expected (H,W), (B,H,W), or (B,1,H,W); got {tuple(images.shape)}.")


def ensure_channel_images(images: torch.Tensor) -> torch.Tensor:
    batched = ensure_batch_images(images)
    return batched.unsqueeze(1)


def gaussian_kernel2d(
    sigma: float,
    *,
    truncate: float = 4.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, 1, 1, 1, device=device, dtype=dtype or torch.float32)
    radius = max(1, int(math.ceil(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype or torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def psf_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    x = ensure_channel_images(images)
    kernel = gaussian_kernel2d(sigma, device=x.device, dtype=x.dtype)
    pad_y = kernel.shape[-2] // 2
    pad_x = kernel.shape[-1] // 2
    x_pad = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    out = F.conv2d(x_pad, kernel)
    return out[:, 0]


def downsample_flux_preserving(images: torch.Tensor, size: int) -> torch.Tensor:
    x = ensure_channel_images(images)
    flux_before = x.sum(dim=(-2, -1), keepdim=True)
    x_ds = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    flux_after = x_ds.sum(dim=(-2, -1), keepdim=True)
    out = x_ds * (flux_before / (flux_after + 1e-12))
    return out[:, 0]


def add_gaussian_noise(
    images: torch.Tensor,
    sigma: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    noise = torch.randn(
        images.shape,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    return images + sigma * noise


def apply_linear_operator(
    images: torch.Tensor,
    A: torch.Tensor,
    *,
    output_shape: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    x = ensure_batch_images(images)
    y = x.reshape(x.shape[0], -1) @ A.T
    if output_shape is None:
        return y
    return y.reshape(x.shape[0], *output_shape)


def build_psf_downsample_matrix(
    image_shape: Sequence[int],
    *,
    sigma_psf: float,
    downsample_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 256,
) -> torch.Tensor:
    height, width = int(image_shape[0]), int(image_shape[1])
    n_in = height * width
    if device is None:
        device = torch.device("cpu")

    basis = torch.eye(n_in, dtype=dtype, device=device).reshape(n_in, height, width)
    outputs: list[torch.Tensor] = []
    for start in range(0, n_in, chunk_size):
        stop = min(start + chunk_size, n_in)
        chunk = basis[start:stop]
        chunk = psf_blur(chunk, sigma_psf)
        chunk = downsample_flux_preserving(chunk, downsample_size)
        outputs.append(chunk.reshape(stop - start, -1))
    return torch.cat(outputs, dim=0).T.contiguous()


def load_fallback_image(path: Path | str, *, size: int = SOURCE_NPIX) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def preprocess_uploaded_image(
    image: Image.Image,
    *,
    size: int = SOURCE_NPIX,
) -> torch.Tensor:
    grayscale = image.convert("L")
    width, height = grayscale.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    grayscale = grayscale.crop((left, top, left + crop, top + crop))
    grayscale = grayscale.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    array = array - array.min()
    array = array / max(array.max(), 1e-6)
    return torch.from_numpy(array)
