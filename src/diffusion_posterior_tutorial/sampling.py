from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass(slots=True)
class HistoryFrame:
    step: int
    time: float
    latent: torch.Tensor
    estimate: torch.Tensor
    forward_projection: torch.Tensor
    residual: torch.Tensor


@dataclass(slots=True)
class PosteriorRunResult:
    samples: torch.Tensor
    observation: torch.Tensor
    history: list[HistoryFrame] = field(default_factory=list)

    def posterior_mean(self) -> torch.Tensor:
        return self.samples.mean(dim=0)

    def posterior_std(self) -> torch.Tensor:
        return self.samples.std(dim=0, unbiased=False)


def _standardize_observations(observation: torch.Tensor, device: torch.device) -> torch.Tensor:
    obs = observation
    if obs.ndim == 2:
        obs = obs.unsqueeze(0).unsqueeze(0)
    elif obs.ndim == 3:
        obs = obs.unsqueeze(1)
    elif obs.ndim == 4 and obs.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unsupported observation shape {tuple(observation.shape)}.")
    return obs.to(device)


def _record_history_frame(
    *,
    history: list[HistoryFrame],
    step: int,
    time_tensor: torch.Tensor,
    latent: torch.Tensor,
    estimate: torch.Tensor,
    forward_projection: torch.Tensor,
    reference_observation: torch.Tensor,
    sigma_n: float,
    max_history_samples: int,
) -> None:
    k = min(max_history_samples, latent.shape[0])
    reference = reference_observation[:1].expand(k, -1, -1, -1)
    residual = (reference - forward_projection[:k]) / sigma_n
    history.append(
        _make_history_frame(
            step=step,
            time_tensor=time_tensor,
            latent=latent[:k],
            estimate=estimate[:k],
            forward_projection=forward_projection[:k],
            reference_observation=reference,
            sigma_n=sigma_n,
        )
    )


def _make_history_frame(
    *,
    step: int,
    time_tensor: torch.Tensor,
    latent: torch.Tensor,
    estimate: torch.Tensor,
    forward_projection: torch.Tensor,
    reference_observation: torch.Tensor,
    sigma_n: float,
) -> HistoryFrame:
    residual = (reference_observation - forward_projection) / sigma_n
    return HistoryFrame(
        step=int(step),
        time=float(time_tensor[0].item()),
        latent=latent.detach().cpu(),
        estimate=estimate.detach().cpu(),
        forward_projection=forward_projection.detach().cpu(),
        residual=residual.detach().cpu(),
    )


class LinearGaussianPosteriorSampler:
    """Teaching-oriented linear-Gaussian posterior sampler with saved history."""

    def __init__(
        self,
        *,
        observation: torch.Tensor,
        A: torch.Tensor,
        model,
        sigma_n: float,
        C: float = 1.0,
        M: float = 0.0,
        anneal_factor: float = 7.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or A.device
        self.model = model.to(self.device) if hasattr(model, "to") else model
        self.obs = _standardize_observations(observation, self.device)
        self.obs_flat = self.obs.flatten(2).squeeze(1)
        self.B = self.obs.shape[0]
        self.Mobs = self.obs_flat.shape[-1]
        self.A = A.to(self.device)
        self.Msrc = self.A.shape[1]
        self.Hs = int(math.isqrt(self.Msrc))
        if self.Hs * self.Hs != self.Msrc:
            raise ValueError("Source dimension is not a perfect square.")
        if self.A.shape[0] != self.Mobs:
            raise ValueError("Observation dimension and operator dimension are inconsistent.")
        self.sigma_n = float(sigma_n)
        self.C = float(C)
        self.M = float(M)
        self.anneal_factor = float(anneal_factor)
        self.diag_AAT = self.A.square().sum(dim=1)

    def _likelihood_score_diag(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x_score = x.detach().clone().requires_grad_(True)
        x_phys = x_score * self.C + self.M
        mean = x_phys.view(n, self.Msrc) @ self.A.T
        mean = mean.unsqueeze(1).expand(-1, self.B, -1)
        y = self.obs_flat.unsqueeze(0).expand(n, -1, -1)
        sigma_t = self.model.sde.sigma(t).view(n, 1, 1)
        rt_2 = sigma_t.square() * ((self.C + self.anneal_factor) ** 2 * t.view(n, 1, 1).pow(4) + 1.0)
        var = self.sigma_n**2 + rt_2 * self.diag_AAT.view(1, 1, self.Mobs)
        diff = y - mean
        ll = -0.5 * (diff.square() / var).sum()
        ll.backward()
        return x_score.grad.view(n, self.Msrc)

    def run(
        self,
        *,
        n_samples: int = 4,
        steps: int = 100,
        corrector_steps: int = 0,
        progress: bool = True,
        record_history: bool = False,
        history_stride: int = 10,
        max_history_samples: int = 2,
        live_callback: Optional[Callable[[HistoryFrame], None]] = None,
        live_stride: int = 5,
    ) -> PosteriorRunResult:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = lambda x: x
            progress = False

        x = self.model.sde.prior((1, self.Hs, self.Hs)).sample([n_samples]).to(self.device)
        t = torch.full((n_samples,), float(self.model.sde.T), device=self.device)
        dt = -(self.model.sde.T - self.model.sde.epsilon) / steps
        history: list[HistoryFrame] = []

        iterator = tqdm(range(steps), disable=not progress)
        for step in iterator:
            t_old = t
            t_new = t + dt

            with torch.no_grad():
                g1 = self.model.sde.diffusion(t_old, x)
                f1 = self.model.sde.drift(t_old, x)
                s1 = self.model.score(t_old, x)
            lk1 = self._likelihood_score_diag(t_old, x).view(n_samples, 1, self.Hs, self.Hs)
            drift1 = f1 - g1.square() * (s1 + lk1)
            dw = torch.randn_like(x) * (-dt) ** 0.5
            x_e = x + drift1 * dt + g1 * dw

            with torch.no_grad():
                g2 = self.model.sde.diffusion(t_new, x_e)
                f2 = self.model.sde.drift(t_new, x_e)
                s2 = self.model.score(t_new, x_e)
            lk2 = self._likelihood_score_diag(t_new, x_e).view(n_samples, 1, self.Hs, self.Hs)
            drift2 = f2 - g2.square() * (s2 + lk2)

            x = x + 0.5 * (drift1 + drift2) * dt + g1 * dw
            x_mean = x - g1 * dw
            t = t_new

            for _ in range(corrector_steps):
                with torch.no_grad():
                    sigma = self.model.sde.sigma(t)
                    score = self.model.score(t, x)
                lk = self._likelihood_score_diag(t, x).view(n_samples, 1, self.Hs, self.Hs)
                eps = (0.05 * sigma).view(-1, 1, 1, 1).square()
                x = x + eps * (score + lk) + (2 * eps).sqrt() * torch.randn_like(x)
                x_mean = x

            should_store = record_history and (
                step == 0 or step == steps - 1 or step % max(1, history_stride) == 0
            )
            should_live = live_callback is not None and (
                step == 0 or step == steps - 1 or step % max(1, live_stride) == 0
            )
            if should_store or should_live:
                with torch.no_grad():
                    estimate = self.model.tweedie(t, x_mean).view(n_samples, 1, self.Hs, self.Hs)
                    estimate_phys = estimate * self.C + self.M
                    forward = (estimate_phys.view(n_samples, self.Msrc) @ self.A.T).view(
                        n_samples,
                        1,
                        self.obs.shape[-2],
                        self.obs.shape[-1],
                    )
                n_keep = min(max_history_samples, n_samples)
                frame = _make_history_frame(
                    step=step,
                    time_tensor=t,
                    latent=x_mean[:n_keep],
                    estimate=estimate_phys[:n_keep],
                    forward_projection=forward[:n_keep],
                    reference_observation=self.obs[:1].expand(n_keep, -1, -1, -1),
                    sigma_n=self.sigma_n,
                )
                if should_store:
                    history.append(frame)
                if should_live:
                    live_callback(frame)

        with torch.no_grad():
            t0 = torch.full((n_samples,), float(self.model.sde.t_min), device=self.device)
            x0 = self.model.tweedie(t0, x_mean).view(n_samples, 1, self.Hs, self.Hs)
            samples = x0 * self.C + self.M
        return PosteriorRunResult(samples=samples, observation=self.obs.detach().cpu(), history=history)


class FixedLensPosteriorSampler(LinearGaussianPosteriorSampler):
    """CLA source sampler for a fixed-lens `caustics` forward model."""
