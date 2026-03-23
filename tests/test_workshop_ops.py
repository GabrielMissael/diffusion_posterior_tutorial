from __future__ import annotations

import torch

from diffusion_posterior_tutorial.workshop_ops import (
    apply_linear_operator,
    build_psf_downsample_matrix,
    downsample_flux_preserving,
    psf_blur,
)


def test_downsample_flux_preserving_matches_total_flux() -> None:
    image = torch.zeros(1, 8, 8)
    image[:, 2:6, 2:6] = 1.0
    out = downsample_flux_preserving(image, 4)
    assert out.shape == (1, 4, 4)
    assert torch.allclose(out.sum(), image.sum(), atol=1e-5, rtol=1e-5)


def test_psf_downsample_matrix_matches_direct_operator() -> None:
    sigma = 0.7
    size = 4
    images = torch.rand(3, 8, 8)
    A = build_psf_downsample_matrix((8, 8), sigma_psf=sigma, downsample_size=size)
    direct = downsample_flux_preserving(psf_blur(images, sigma), size)
    linear = apply_linear_operator(images, A, output_shape=(size, size))
    assert A.shape == (size * size, 64)
    assert torch.allclose(linear, direct, atol=2e-4, rtol=2e-4)
