from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.func import vmap

from .workshop_ops import OBS_NPIX, SOURCE_NPIX, add_gaussian_noise, get_default_device, psf_blur

try:
    import caustics
    from caustics import Module, forward
except ImportError:  # pragma: no cover - exercised via importorskip tests
    caustics = None
    Module = object

    def forward(fn):  # type: ignore[misc]
        return fn


def caustics_is_available() -> bool:
    return caustics is not None


def _require_caustics() -> None:
    if caustics is None:
        raise ImportError(
            "The `caustics` package is required for the lensing tutorial. "
            "Install `caustics==1.2.0` before running this section."
        )


@dataclass(slots=True)
class LensingSceneConfig:
    theta_e: float = 1.35
    phi: float = 0.4
    axis_ratio: float = 0.82
    source_x: float = 0.08
    source_y: float = -0.05
    lens_x: float = 0.0
    lens_y: float = 0.0
    slope_t: float = 1.0
    psf_fwhm: float = 0.18
    noise_sigma: float = 0.03


class TutorialLensingSimulator(Module):
    """Minimal `caustics`-based forward model for source-plane inference."""

    def __init__(
        self,
        *,
        source_pixels: int = SOURCE_NPIX,
        observation_pixels: int = OBS_NPIX,
        source_fov: float = 6.0,
        observation_fov: float = 6.0,
        z_l: float = 0.5,
        z_s: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        _require_caustics()
        if source_pixels != SOURCE_NPIX or observation_pixels != OBS_NPIX:
            raise ValueError("The tutorial simulator is fixed to 64x64 source and observation grids.")

        self.device = device or get_default_device()
        self.dtype = dtype
        super().__init__("tutorial_lensing")

        self.source_pixels = source_pixels
        self.observation_pixels = observation_pixels
        self.source_fov = float(source_fov)
        self.observation_fov = float(observation_fov)
        self.source_pixelscale = self.source_fov / self.source_pixels
        self.observation_pixelscale = self.observation_fov / self.observation_pixels
        self.source_offset_x = 0.0
        self.source_offset_y = 0.0
        self._matrix_cache: dict[tuple[float, ...], torch.Tensor] = {}

        self.cosmo = caustics.FlatLambdaCDM(name="cosmo")
        self.lens_epl = caustics.EPL(cosmology=self.cosmo, name="epl")
        self.lens = caustics.SinglePlane(name="lensmass", cosmology=self.cosmo, lenses=[self.lens_epl])
        self.source = caustics.Pixelated(
            pixelscale=self.source_pixelscale,
            shape=(self.source_pixels, self.source_pixels),
            x0=0.0,
            y0=0.0,
            name="source",
        )
        self.lens.z_l = z_l
        self.lens.z_s = z_s

        thx, thy = caustics.utils.meshgrid(
            self.observation_pixelscale,
            self.observation_pixels,
            dtype=dtype,
            device=self.device,
        )
        self.thx = thx.to(self.device)
        self.thy = thy.to(self.device)
        self.to(device=self.device, dtype=dtype)

    @forward
    def __call__(self):
        bx, by = self.lens.raytrace(self.thx, self.thy)
        image = self.source.brightness(
            bx - self.source_offset_x,
            by - self.source_offset_y,
        )
        return image

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) >= 1 and isinstance(args[0], (str, torch.device)):
            device = args[0]
        if len(args) >= 1 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if len(args) >= 2 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        if isinstance(self.thx, torch.Tensor):
            self.thx = self.thx.to(device=self.device, dtype=self.dtype)
            self.thy = self.thy.to(device=self.device, dtype=self.dtype)
        return self

    def parameter_vector(self, config: LensingSceneConfig) -> torch.Tensor:
        values = [
            float(config.lens_x),
            float(config.lens_y),
            float(config.axis_ratio),
            float(config.phi),
            float(config.theta_e),
            float(config.slope_t),
        ]
        return torch.tensor(values, device=self.device, dtype=self.dtype)

    def _prepare_source(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim == 2:
            source = source.unsqueeze(0)
        elif source.ndim == 4 and source.shape[1] == 1:
            source = source[:, 0]
        if source.ndim != 3:
            raise ValueError(f"Expected source shape (H,W), (B,H,W), or (B,1,H,W); got {tuple(source.shape)}.")
        if source.shape[-2:] != (self.source_pixels, self.source_pixels):
            raise ValueError(
                f"Expected source images with shape ({self.source_pixels}, {self.source_pixels}); "
                f"got {tuple(source.shape[-2:])}."
            )
        return source.to(device=self.device, dtype=self.dtype)

    def simulate_clean(self, source: torch.Tensor, config: LensingSceneConfig) -> torch.Tensor:
        source_batch = self._prepare_source(source)
        self.source_offset_x = float(config.source_x)
        self.source_offset_y = float(config.source_y)
        lens_params = self.parameter_vector(config).unsqueeze(0).expand(source_batch.shape[0], -1)
        source_flat = source_batch.reshape(source_batch.shape[0], -1)
        x_fwd = torch.cat([lens_params, source_flat], dim=-1)
        output = vmap(self.__call__)(x_fwd).reshape(source_batch.shape[0], self.observation_pixels, self.observation_pixels)
        sigma_arcsec = float(config.psf_fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        sigma_pixels = sigma_arcsec / self.observation_pixelscale
        if sigma_pixels > 0:
            output = psf_blur(output, sigma_pixels)
        return output

    def _matrix_cache_key(self, config: LensingSceneConfig) -> tuple[float, ...]:
        return (
            round(float(config.theta_e), 6),
            round(float(config.phi), 6),
            round(float(config.axis_ratio), 6),
            round(float(config.source_x), 6),
            round(float(config.source_y), 6),
            round(float(config.lens_x), 6),
            round(float(config.lens_y), 6),
            round(float(config.slope_t), 6),
            round(float(config.psf_fwhm), 6),
        )

    def build_matrix(
        self,
        config: LensingSceneConfig,
        *,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        key = self._matrix_cache_key(config)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached

        n_in = self.source_pixels * self.source_pixels
        outputs: list[torch.Tensor] = []
        basis = torch.eye(n_in, device=self.device, dtype=self.dtype).reshape(
            n_in, self.source_pixels, self.source_pixels
        )
        for start in range(0, n_in, chunk_size):
            stop = min(start + chunk_size, n_in)
            out = self.simulate_clean(basis[start:stop], config)
            outputs.append(out.reshape(stop - start, -1))
        matrix = torch.cat(outputs, dim=0).T.contiguous()
        self._matrix_cache[key] = matrix
        return matrix

    def observe(
        self,
        source: torch.Tensor,
        config: LensingSceneConfig,
        *,
        generator: Optional[torch.Generator] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        clean = self.simulate_clean(source, config)
        if not add_noise:
            return clean
        return add_gaussian_noise(clean, float(config.noise_sigma), generator=generator)
