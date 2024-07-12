import rootutils

rootutils.setup_root(__file__, indicator=".env", pythonpath=True)
import hydra
import torch
from typing import Optional
from omegaconf import DictConfig
from src.trainer.task import train_task
from src.utils import utils, pylogger

log = pylogger.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def train(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    use_cudnn = cfg.get("use_cudnn_benchmark", False)
    if use_cudnn:
        torch.backends.cudnn.benchmark = True
        log.info(f"Using cudnn.benchmark.")
    else:
        torch.backends.cudnn.benchmark = False
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator

        # torch.cuda.memory._get_current_allocator()
        # TODO: use rmm allocator to speed up memory allocation
        #   https://github.com/rapidsai/rmm/issues/1362
        #   https://github.com/pytorch/pytorch/issues/111366
        rmm.reinitialize(pool_allocator=True)
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        log.info(f"Unable the RMM allocator.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)

    metric_dict, _ = train_task(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


def main():
    train()


if __name__ == "__main__":
    main()
