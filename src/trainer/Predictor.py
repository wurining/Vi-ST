import rootutils

rootutils.setup_root(__file__, indicator=".env", pythonpath=True)
import hydra
import torch
from omegaconf import DictConfig
from src.trainer.task import eval_task
from src.utils import utils, pylogger

log = pylogger.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval.yaml")
def eval(cfg: DictConfig) -> None:
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

    eval_task(cfg)


def main():
    eval()


if __name__ == "__main__":
    main()
