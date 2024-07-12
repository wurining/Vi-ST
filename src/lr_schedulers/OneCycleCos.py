from torch.optim.lr_scheduler import (
    SequentialLR,
    CosineAnnealingLR,
    LinearLR,
)

from torch.optim.optimizer import Optimizer


class OneCycleCos(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_min: float,
        milestones: int,
        T_max: int,
        lr_max: float = None,
        lr_start: float = None,
        verbose: bool = False,
    ) -> None:
        start_factor = 0.1
        if lr_start is not None and lr_max is not None:
            start_factor = lr_start / lr_max
        schedulers = [
            LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1,
                total_iters=milestones,
                verbose=verbose,
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=T_max - milestones,
                eta_min=lr_min,
                last_epoch=-1,
                verbose=verbose,
            ),
        ]

        super().__init__(optimizer, schedulers, [milestones], -1, verbose)

    def step(self, *args, **kwargs):
        super().step()

    def state_dict(self):
        return super().state_dict()

    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)
