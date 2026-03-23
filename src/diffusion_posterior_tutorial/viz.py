from __future__ import annotations

from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .sampling import HistoryFrame, PosteriorRunResult


def tensor_to_display(
    image: torch.Tensor,
    *,
    log_scale: bool = False,
    min_val: float = 1e-2,
) -> np.ndarray:
    array = image.detach().cpu().float().squeeze().numpy()
    if log_scale:
        array = np.log(np.clip(array, min_val, None))
    finite = np.isfinite(array)
    if finite.any():
        lo = array[finite].min()
        hi = array[finite].max()
        if hi > lo:
            array = (array - lo) / (hi - lo)
        else:
            array = np.zeros_like(array)
    else:
        array = np.zeros_like(array)
    array[~finite] = 0.0
    return array


def plot_image_grid(
    images: Sequence[torch.Tensor],
    *,
    titles: Optional[Sequence[str]] = None,
    cmap: str = "magma",
    log_scale: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    origin: str = "upper",
) -> plt.Figure:
    n = len(images)
    figsize = figsize or (3 * n, 3)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    for idx, image in enumerate(images):
        ax = axes[0, idx]
        ax.imshow(tensor_to_display(image, log_scale=log_scale), cmap=cmap, origin=origin)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[idx])
    fig.tight_layout()
    return fig


def plot_posterior_summary(
    result: PosteriorRunResult,
    *,
    truth: Optional[torch.Tensor] = None,
    log_scale: bool = True,
    origin: str = "upper",
) -> plt.Figure:
    mean = result.posterior_mean()[0]
    std = result.posterior_std()[0]
    images = []
    titles = []
    if truth is not None:
        images.append(truth)
        titles.append("Reference")
    images.extend([mean, std])
    titles.extend(["Posterior mean", "Posterior std"])
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 3.5), squeeze=False)
    for idx, image in enumerate(images):
        ax = axes[0, idx]
        use_log = log_scale and idx != len(images) - 1
        ax.imshow(tensor_to_display(image, log_scale=use_log), cmap="magma", origin=origin)
        ax.axis("off")
        ax.set_title(titles[idx])
    fig.tight_layout()
    return fig


def plot_forward_residual_panel(
    samples: torch.Tensor,
    *,
    observation: torch.Tensor,
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    noise_sigma: float,
    max_show: int = 4,
    origin: str = "upper",
) -> plt.Figure:
    obs = observation.detach().cpu().squeeze()
    n_show = min(max_show, samples.shape[0])
    forward = forward_model(samples[:n_show, 0]).detach().cpu()
    if n_show == 1:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), squeeze=False)
        axes[0, 0].imshow(tensor_to_display(samples[0, 0], log_scale=True), cmap="magma", origin=origin)
        axes[0, 0].set_title("Posterior sample")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(tensor_to_display(forward[0], log_scale=False), cmap="magma", origin=origin)
        axes[0, 1].set_title("Forward projection")
        axes[0, 1].axis("off")

        residual = (obs - forward[0]) / noise_sigma
        axes[0, 2].imshow(residual, cmap="coolwarm", origin=origin, vmin=-4, vmax=4)
        axes[0, 2].set_title("Normalized residual")
        axes[0, 2].axis("off")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 8), squeeze=False)
    for idx in range(n_show):
        axes[0, idx].imshow(tensor_to_display(samples[idx, 0], log_scale=True), cmap="magma", origin=origin)
        axes[0, idx].set_title(f"Sample {idx + 1}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(tensor_to_display(forward[idx], log_scale=False), cmap="magma", origin=origin)
        axes[1, idx].set_title("Forward projection")
        axes[1, idx].axis("off")

        residual = (obs - forward[idx]) / noise_sigma
        axes[2, idx].imshow(residual, cmap="coolwarm", origin=origin, vmin=-4, vmax=4)
        axes[2, idx].set_title("Normalized residual")
        axes[2, idx].axis("off")
    fig.tight_layout()
    return fig


def _frame_indices(history: Sequence[HistoryFrame], max_frames: int) -> list[int]:
    if len(history) <= max_frames:
        return list(range(len(history)))
    return np.linspace(0, len(history) - 1, num=max_frames, dtype=int).tolist()


def plot_history_frame(
    frame: HistoryFrame,
    *,
    sample_index: int = 0,
    log_source: bool = True,
    origin: str = "upper",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), squeeze=False)
    panels = [
        (frame.latent[sample_index], "Current state", "magma", False),
        (frame.estimate[sample_index], "Denoised estimate", "magma", log_source),
        (frame.forward_projection[sample_index], "Forward projection", "magma", False),
        (frame.residual[sample_index], "Residual / noise", "coolwarm", False),
    ]
    for idx, (image, title, cmap, use_log) in enumerate(panels):
        ax = axes[0, idx]
        if title == "Residual / noise":
            ax.imshow(image.squeeze(), cmap=cmap, origin=origin, vmin=-4, vmax=4)
        else:
            ax.imshow(tensor_to_display(image, log_scale=use_log), cmap=cmap, origin=origin)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Step {frame.step}, t = {frame.time:.3f}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_history_montage(
    history: Sequence[HistoryFrame],
    *,
    sample_index: int = 0,
    max_frames: int = 6,
    log_source: bool = True,
    origin: str = "upper",
) -> plt.Figure:
    indices = _frame_indices(history, max_frames)
    if not indices:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.text(0.5, 0.5, "No history recorded.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, len(indices), figsize=(2.8 * len(indices), 3.2), squeeze=False)
    for col, history_index in enumerate(indices):
        frame = history[history_index]
        ax = axes[0, col]
        ax.imshow(
            tensor_to_display(frame.estimate[sample_index], log_scale=log_source),
            cmap="magma",
            origin=origin,
        )
        ax.axis("off")
        ax.set_title(f"Step {frame.step}\nt={frame.time:.2f}")
    fig.tight_layout()
    return fig


def build_history_scrubber(
    history: Sequence[HistoryFrame],
    *,
    sample_index: int = 0,
    log_source: bool = True,
    origin: str = "upper",
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the history scrubber.") from exc

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(len(history) - 1, 0),
        step=1,
        description="History frame",
        continuous_update=False,
        layout=widgets.Layout(width="700px"),
    )
    output = widgets.Output()

    def _draw(change=None):
        del change
        with output:
            output.clear_output(wait=True)
            if not history:
                print("No history was recorded.")
                return
            fig = plot_history_frame(
                history[slider.value],
                sample_index=sample_index,
                log_source=log_source,
                origin=origin,
            )
            display(fig)
            plt.close(fig)

    slider.observe(_draw, names="value")
    _draw()
    return widgets.VBox([slider, output])


def render_live_frame(
    frame: HistoryFrame,
    *,
    sample_index: int = 0,
    log_source: bool = True,
    origin: str = "upper",
) -> plt.Figure:
    return plot_history_frame(
        frame,
        sample_index=sample_index,
        log_source=log_source,
        origin=origin,
    )
