from typing import Any, Dict, Tuple
from matplotlib.ticker import MultipleLocator
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from src.models.components.eccv import ViST
from src.loss.ViSTloss import ViSTLoss
from src.metrics.corr import pearson_corr_coef
from tslearn.metrics import soft_dtw


class ViSTModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        featrue_key: str,
        index_start: int,
        index_length: int,
        indeces: list,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.featrue_key = featrue_key
        self.index_start = index_start if index_start is not None else 0
        assert index_length is not None, "index_length is required"
        self.index_length = index_length
        self.indeces_range = slice(
            self.index_start, self.index_start + self.index_length
        )

        self.indeces = list(
            range(self.index_start, self.index_start + self.index_length)
        )
        if indeces is not None:
            assert self.index_length <= len(indeces), "indeces length is not match"
            self.indeces = indeces

        self.configure_loss()
        self.configure_metrics()

    def configure_loss(self):
        """
        Define loss function
        """
        self.loss = ViSTLoss()

    def configure_metrics(self):
        self.corr_metric = pearson_corr_coef
        self.dtw_metric = soft_dtw

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(*x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        y = batch["spike_avg"][:, :, self.indeces[self.indeces_range]]  # spike_avg_std
        x = batch[self.featrue_key]
        if not self.featrue_key.startswith("dinov2_feats_"):
            raise NotImplementedError("The feature key is not implemented yet.")
        if not isinstance(self.net, ViST):
            raise NotImplementedError(
                f"This model is not implemented yet: {self.net.__class__.__name__}"
            )
        y_hat = self.forward(
            x, batch["rf_to_dinov2"][:, self.indeces[self.indeces_range], :, :]
        )
        loss = self.loss(y_hat, y)
        y = y.permute(0, 2, 1)  # N x n_neurons x n_frames
        return loss, y_hat, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        _, corr = self.corr_metric(preds, targets)
        dtw_score = self.dtw_metric(
            rearrange(preds, "n c t -> t (n c)"), rearrange(targets, "n c t -> t (n c)")
        )

        # update and log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/corr", corr, on_step=False, on_epoch=True)
        self.log("train/dtw_score", dtw_score, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        _, corr = self.corr_metric(preds, targets)
        dtw_score = self.dtw_metric(
            rearrange(preds, "n c t -> t (n c)"), rearrange(targets, "n c t -> t (n c)")
        )

        # update and log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/corr", corr, on_step=False, on_epoch=True)
        self.log("val/dtw_score", dtw_score, on_step=False, on_epoch=True)

    def _plot_responses(self, preds, targets, title=None):
        # preds: N x n_neurons x n_frames
        # targets: N x n_neurons x n_frames
        cell = preds.shape[1]
        preds = preds.reshape(-1, cell).cpu().float().numpy()
        targets = targets.reshape(-1, cell).cpu().float().numpy()

        # generate line plot
        fig, axs = plt.subplots(cell, 1, figsize=(10, cell * 1.5))
        fig.set_dpi(240)
        if cell == 1:
            axs = [axs]
        for i in range(cell):
            axs[i].set_title(f"neuron {i}", loc="left")
            axs[i].plot(
                preds[:, i],
                label="hat",
                color="red",
                alpha=0.4,
                linewidth=0.5,
            )
            axs[i].plot(
                targets[:, i],
                label="gt",
                color="blue",
                alpha=0.4,
                linewidth=0.5,
            )
            axs[i].legend()
        fig.tight_layout()
        self.loggers[1].experiment.add_figure(
            f"preds_{title}", fig, global_step=self.global_step
        )
        fig.savefig(
            f"{self.loggers[1].save_dir}/preds_{title}.png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def _plot_corr(self, corrs, title=None):
        # generate line plot
        # corrs: n_neurons
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        fig.set_dpi(120)
        axs.set_title(f"corr", loc="left")
        axs.set_xlabel("neuron")
        axs.set_ylabel("corr")
        # axs.plot(
        #     np.arange(len(corrs)),
        #     corrs.cpu().float().numpy(),
        #     label="corr",
        #     color="red",
        #     alpha=0.9,
        #     linewidth=0.5,
        # )
        # to scatter
        axs.scatter(
            np.arange(len(corrs)),
            corrs.cpu().float().numpy(),
            label="corr",
            color="red",
            alpha=0.9,
            marker=">",
            s=100,
        )
        axs.xaxis.set_major_locator(MultipleLocator(10))
        axs.xaxis.set_minor_locator(MultipleLocator(1))
        axs.tick_params(which="both", width=1)
        axs.tick_params(which="major", length=7)
        axs.tick_params(which="minor", length=4, color="r")
        axs.legend()
        fig.tight_layout()
        self.loggers[1].experiment.add_figure(
            f"corrs_{title}", fig, global_step=self.global_step
        )
        fig.savefig(
            f"{self.loggers[1].save_dir}/preds_{title}.png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, targets = self.model_step(batch)
        corrs, corr = self.corr_metric(preds, targets)
        corrs_modify, corr_modify = self.modify_corr_metric(preds, targets)

        # save each neurons' corr and corr_modify
        try:
            self._plot_corr(corrs, title=f"corrs")
            self._plot_corr(corrs_modify, title=f"corrs_mo")
        except Exception as e:
            print(e)
        # plot predictions and targets
        self._plot_responses(preds, targets, title=f"preds")

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        loss, preds, targets = self.model_step(batch)
        corrs, corr = self.corr_metric(preds, targets)
        corrs_modify, corr_modify = self.modify_corr_metric(preds, targets)
        # calc metrics
        # avg cc
        # corrs = corrs.abs()
        mean_cc = corrs[corrs > 0].float().mean().item()
        std_cc = corrs[corrs > 0].float().std().item()
        save_predications = {
            "preds": preds.cpu().float().numpy(),
            "targets": targets.cpu().float().numpy(),
            "rmse": loss.item(),
            "corrs": corrs.cpu().float().numpy(),
            "mean_cc": mean_cc,
            "std_cc": std_cc,
        }
        return save_predications

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
