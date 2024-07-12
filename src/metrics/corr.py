import torch
import src.utils.pylogger as pylogger
from torchmetrics.functional.regression import pearson_corrcoef


log = pylogger.get_pylogger(__name__)


# function version
def modified_pearson_corr_coef(preds: torch.Tensor, target: torch.Tensor):
    """
    Pearson Correlation Coefficient

    Args:
        preds (torch.Tensor): predicted responses, N x n_neurons x n_frames
        target (torch.Tensor): ground truth responses, N x n_neurons x n_frames
    Returns:
        single_corr (torch.Tensor): single-trial correlation
        entire_corr (torch.Tensor): entire-trial correlation
    """
    assert preds.shape == target.shape, "preds and target must have the same shape"
    axis = 0
    eps = 1e-8
    preds = torch.cat([*preds], 1).permute(1, 0).float().detach().cpu()
    target = torch.cat([*target], 1).permute(1, 0).float().detach().cpu()
    y1 = (preds - preds.mean(axis=axis, keepdim=True)) / (
        preds.std(axis=axis, keepdim=True, unbiased=False) + eps
    )
    y2 = (target - target.mean(axis=axis, keepdim=True)) / (
        target.std(axis=axis, keepdim=True, unbiased=False) + eps
    )
    ret = (y1 * y2).mean(axis=axis)
    ret[torch.isnan(ret)] = 0
    return ret, ret.mean()


# function version
def pearson_corr_coef(preds: torch.Tensor, target: torch.Tensor):
    """
    Pearson Correlation Coefficient

    Args:
        preds (torch.Tensor): predicted responses, N x n_neurons x n_frames
        target (torch.Tensor): ground truth responses, N x n_neurons x n_frames
    Returns:
        single_corr (torch.Tensor): single-trial correlation
        entire_corr (torch.Tensor): entire-trial correlation
    """
    assert preds.shape == target.shape, "preds and target must have the same shape"
    if len(preds.shape) == 3:
        # preds should be AllTimesStep x n_neurons (axis=0)
        preds = torch.cat([*preds], 1).permute(1, 0)
        target = torch.cat([*target], 1).permute(1, 0)

    ret = pearson_corrcoef(preds, target)
    # if torch.any(torch.isnan(ret)):
    #     log.warning(f"NaNs will be set to Zero.")
    ret[torch.isnan(ret)] = 0
    return ret, ret.abs().mean()
