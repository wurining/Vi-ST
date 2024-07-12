import torch
import torch.nn as nn
from tslearn.metrics import SoftDTWLossPyTorch


class ViSTLoss(nn.Module):
    def __init__(self):
        super(ViSTLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.dtw_loss = SoftDTWLossPyTorch(gamma=0.1, normalize=True)

    def forward(self, y_hat, y):
        y_hat = y_hat.permute(0, 2, 1)
        y_shape = y.size()
        # rmse
        loss = 0.1 * torch.sqrt(self.mse(y_hat, y))
        # -relu
        loss += 0.5 * self.relu(-y_hat).mean()
        # SoftDTW 6
        window_size = y_shape[1] // 20
        y_hat_batch = torch.stack(
            [
                y_hat[:, i : i + window_size, :]
                for i in range(0, y_shape[1] - window_size, window_size // 2)
            ],
            dim=0,
        ).reshape(-1, window_size, y_shape[-1])
        y_b = torch.stack(
            [
                y[:, i : i + window_size, :]
                for i in range(0, y_shape[1] - window_size, window_size // 2)
            ],
            dim=0,
        ).reshape(-1, window_size, y_shape[-1])
        loss += 0.5 * (self.dtw_loss(y_hat_batch, y_b).mean() * 1e-5)
        # SoftDTW 12
        window_size = y_shape[1] // 10
        y_hat_batch = torch.stack(
            [
                y_hat[:, i : i + window_size, :]
                for i in range(0, y_shape[1] - window_size, window_size // 2)
            ],
            dim=0,
        ).reshape(-1, window_size, y_shape[-1])
        y_b = torch.stack(
            [
                y[:, i : i + window_size, :]
                for i in range(0, y_shape[1] - window_size, window_size // 2)
            ],
            dim=0,
        ).reshape(-1, window_size, y_shape[-1])
        loss += 0.5 * (self.dtw_loss(y_hat_batch, y_b).mean() * 1e-5)
        return loss
