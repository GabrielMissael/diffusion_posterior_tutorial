from __future__ import annotations

import torch

from diffusion_posterior_tutorial.sampling import FixedLensPosteriorSampler, LinearGaussianPosteriorSampler


class _DummyPrior:
    def __init__(self, sample_shape: tuple[int, ...]) -> None:
        self.sample_shape = sample_shape

    def sample(self, batch_shape: list[int]) -> torch.Tensor:
        return torch.randn(*batch_shape, *self.sample_shape)


class _DummySDE:
    T = 1.0
    epsilon = 1e-2
    t_min = 1e-3

    def prior(self, shape: tuple[int, ...]) -> _DummyPrior:
        return _DummyPrior(shape)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return 0.4 * torch.ones_like(t)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return 0.2 * torch.ones_like(x)

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return -0.05 * x


class _DummyModel:
    def __init__(self) -> None:
        self.sde = _DummySDE()

    def to(self, device: torch.device):
        self.device = device
        return self

    def score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return -0.1 * x

    def tweedie(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + t.view(-1, 1, 1, 1))


def test_linear_sampler_records_history() -> None:
    model = _DummyModel()
    observation = torch.zeros(4, 4)
    A = torch.eye(16)
    sampler = LinearGaussianPosteriorSampler(
        observation=observation,
        A=A,
        model=model,
        sigma_n=0.1,
        device=torch.device("cpu"),
    )
    result = sampler.run(n_samples=2, steps=6, progress=False, record_history=True, history_stride=2)
    assert result.samples.shape == (2, 1, 4, 4)
    assert len(result.history) >= 3
    assert result.history[0].estimate.shape[0] <= 2


def test_fixed_lens_sampler_records_history() -> None:
    model = _DummyModel()
    observation = torch.zeros(4, 4)
    A = torch.eye(16)
    sampler = FixedLensPosteriorSampler(
        observation=observation,
        A=A,
        model=model,
        sigma_n=0.1,
        device=torch.device("cpu"),
    )
    result = sampler.run(n_samples=2, steps=5, progress=False, record_history=True, history_stride=2)
    assert result.samples.shape == (2, 1, 4, 4)
    assert len(result.history) >= 3
    assert result.history[-1].forward_projection.shape[-2:] == (4, 4)
