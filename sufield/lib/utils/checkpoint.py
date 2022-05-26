import os
import torch

from .distributed import get_rank


def checkpoint(args: dict,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               step: int,
               metrics: dict,
               scaler: torch.cuda.amp.GradScaler,
               suffix: str = None):
    if get_rank() == 0:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step,
            'metrics': metrics,
            'scaler': scaler.state_dict(),
            'args': args,
        }
        prefix = f'step#{step}' if suffix is None else suffix
        os.makedirs(f"{args['output_dir']}/checkpoints/", exist_ok=True)
        torch.save(state_dict, f"{args['output_dir']}/checkpoints/checkpoint-{prefix}.pth")
