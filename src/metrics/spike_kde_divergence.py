import torch
import numpy as np
import src.utils.pylogger as pylogger
from scipy.signal import find_peaks, peak_widths
from scipy.stats import gaussian_kde
from torchmetrics.functional.regression import kl_divergence

log = pylogger.get_pylogger(__name__)


def _define_support_grid(x, bw, cut, clip, gridsize):
    """Create the grid of evaluation points depending for vector x."""
    clip_lo = -np.inf if clip[0] is None else clip[0]
    clip_hi = +np.inf if clip[1] is None else clip[1]
    gridmin = max(x.min() - bw * cut, clip_lo)
    gridmax = min(x.max() + bw * cut, clip_hi)
    return np.linspace(gridmin, gridmax, gridsize)


def _define_support_univariate(x, bandwidth_factor=0.5):
    """Create a 1D grid of evaluation points."""
    kde = gaussian_kde(x)
    kde.set_bandwidth(kde.factor * bandwidth_factor)
    bw = np.sqrt(kde.covariance.squeeze())
    grid = _define_support_grid(x, bw, cut=3, clip=(None, None), gridsize=200)
    return grid


def kde_compute(data, bandwidth_factor=0.5, interction_data=None):
    assert interction_data is not None
    grid = _define_support_univariate(
        np.concatenate([data, interction_data]), bandwidth_factor=bandwidth_factor
    )
    kde = gaussian_kde(data, bw_method=bandwidth_factor)
    kde.set_bandwidth(bw_method=kde.factor * bandwidth_factor)
    density_estimate = kde(grid)
    return density_estimate, grid, kde


def calc_kl_divergence(d1, d2):
    return kl_divergence(d1, d2)


def calc_tvd(d1, d2):
    # Total Variation Distance
    return 0.5 * np.sum(np.abs(d1 - d2))


def cells_peak_widths(cell_spikes, high_cut=1.0):
    x = cell_spikes
    x_ = np.array(cell_spikes)
    x_[x_ > high_cut] = high_cut
    peaks, _ = find_peaks(x, height=0)
    return peak_widths(x_, peaks, rel_height=high_cut)[0]


def calc_scores(y, y_hat, bandwidth_factor=0.5, high_cut=1.0, divergence="kl"):
    y = cells_peak_widths(y, high_cut=high_cut)
    y_hat = cells_peak_widths(y_hat, high_cut=high_cut)
    if len(y) < 2 or len(y_hat) < 2:
        return np.inf

    try:
        density_y, grid_y, kde_y = kde_compute(
            y, bandwidth_factor=bandwidth_factor, interction_data=y_hat
        )
        density_y_hat, grid_y_hat, kde_y_hat = kde_compute(
            y_hat, bandwidth_factor=bandwidth_factor, interction_data=y
        )
        if divergence == "kl":
            score = calc_kl_divergence(
                torch.tensor([density_y_hat]), torch.tensor([density_y])
            )
        elif divergence == "tvd":
            score = calc_tvd(density_y_hat, density_y)
        else:
            raise ValueError(f"Unknown divergence method: {divergence}")
    except Exception as e:
        if "singular data" in str(e):
            # log.error(f"Error: {e}")
            score = np.inf
        else:
            raise e
    return score


# function version
# eg. calc_scores(y, y_hat, bandwidth_factor=0.25, high_cut=1.0)
def spike_kde_divergence(
    preds: torch.Tensor,
    target: torch.Tensor,
    divergence: str = "kl",
    bandwidth_factor: float = 0.5,
    high_cut: float = 1.0,
    cut_negative: bool = True,
):
    """
    Spike KDE Divergence

    Args:
        preds (torch.Tensor): predicted responses, N x n_neurons x n_frames
        target (torch.Tensor): ground truth responses, N x n_neurons x n_frames
        divergence (str): divergence algorithm:
            - "kl" (Kullback-Leibler Divergence)
            - "tvd" (Total Variation Distance)
        high_cut (float): threshold for ensuring effective firing rate
    Returns:
        torch.Tensor: mean divergence score
        torch.Tensor: divergence scores for each neuron
    """
    assert preds.shape == target.shape, "preds and target must have the same shape"
    assert (
        len(preds.shape) == 2 or len(preds.shape) == 3
    ), "preds and target must have 2 or 3 dimensions"

    if cut_negative:
        preds[preds < 0] = 0
        target[target < 0] = 0

    if len(preds.shape) == 3:
        # reshape to AllTimesStep x n_neurons
        preds = torch.cat([*preds], 1)
        target = torch.cat([*target], 1)

    scores = [[]] * preds.shape[0]
    for i, (pred, tar) in enumerate(zip(preds, target)):
        scores[i] = calc_scores(
            pred.cpu().numpy(),
            tar.cpu().numpy(),
            bandwidth_factor=bandwidth_factor,
            high_cut=high_cut,
            divergence=divergence,
        )

    scores = np.nan_to_num(scores, nan=1000.0, posinf=1000.0, neginf=0.0)
    return torch.tensor(scores).mean(), torch.tensor(scores)
