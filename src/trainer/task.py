import logging
from pathlib import Path
import pickle
import hydra
import torch
import lightning as L
from typing import List, Tuple
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import src.utils as utils
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@utils.task_wrapper
def train_task(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # allow low precision calculation in tf32 metrix multiplication
    torch.set_float32_matmul_precision("medium")
    # all low precision calculation in conv calculation
    torch.backends.cudnn.allow_tf32 = True

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        # train_dataloaders=train_loader,
        # val_dataloaders=val_loader,
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "data": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        # ckpt_path = trainer.checkpoint_callback.best_model_path
        ckpt_path = trainer.checkpoint_callback.last_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        log.info(f"Best ckpt path: {ckpt_path}")
        # export predictions
        predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        # save predictions
        output_file = Path(ckpt_path).parent.parent / "predictions.pkl"
        print("Saving predictions...")
        # save dict as pickle
        with open(output_file, "wb") as f:
            pickle.dump(predictions, f)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@utils.task_wrapper
def eval_task(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    # allow low precision calculation in tf32 metrix multiplication
    torch.set_float32_matmul_precision("medium")
    # all low precision calculation in conv calculation
    torch.backends.cudnn.allow_tf32 = True

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "data": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # TODO: for predictions use trainer.predict(...)
    predictions = trainer.predict(
        model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path
    )

    # save predictions
    output_file = Path(cfg.ckpt_path).parent.parent / "predictions.pkl"
    print("Saving predictions...")
    # save dict as pickle
    with open(output_file, "wb") as f:
        pickle.dump(predictions, f)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict
