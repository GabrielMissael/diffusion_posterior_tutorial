from __future__ import annotations

import pytest
import torch

pytest.importorskip("caustics")

from diffusion_posterior_tutorial.lensing import LensingSceneConfig, TutorialLensingSimulator


def test_tutorial_lensing_simulator_outputs_64_square_images() -> None:
    simulator = TutorialLensingSimulator(device=torch.device("cpu"))
    source = torch.zeros(1, 64, 64)
    source[:, 28:36, 28:36] = 1.0
    config = LensingSceneConfig()

    clean = simulator.simulate_clean(source, config)
    noisy = simulator.observe(source, config, add_noise=False)

    assert clean.shape == (1, 64, 64)
    assert noisy.shape == (1, 64, 64)


def test_source_position_changes_the_simulated_lensed_image() -> None:
    simulator = TutorialLensingSimulator(device=torch.device("cpu"))
    source = torch.zeros(1, 64, 64)
    source[:, 28:36, 28:36] = 1.0
    centered = simulator.simulate_clean(source, LensingSceneConfig(source_x=0.0, source_y=0.0))
    shifted = simulator.simulate_clean(source, LensingSceneConfig(source_x=0.18, source_y=-0.12))
    assert not torch.allclose(centered, shifted)
