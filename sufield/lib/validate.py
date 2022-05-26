from logging import getLogger

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils.criteria import PerClassCriterion


def validate_pass(model: nn.Module, dataloader: DataLoader, writer: SummaryWriter = None, step: int = 0, logging: bool = True):
    model_status = model.training
    model.eval()
    logger = getLogger(__name__)
    evaluator = PerClassCriterion()
    for step_idx, sample in enumerate(dataloader):
        if logging:
            logger.debug(f"Validate step #{step_idx}")
        prediction, target = model(sample)
        if logging:
            logger.debug(f"Validate step #{step_idx} finish")
        evaluator.update(prediction, target)
    model.train(model_status)
    iou = evaluator.get_iou() * 100
    precision = evaluator.get_precision() * 100
    recall = evaluator.get_recall() * 100
    if logging:
        logger.info("Validation result")
        logger.info("Per class iou:\t" + '\t'.join([f'{i:.2f}' for i in iou]))
        logger.info("Per class prec:\t" + '\t'.join([f'{i:.2f}' for i in precision]))
        logger.info("Per class recall:\t" + '\t'.join([f'{i:.2f}' for i in recall]))
        logger.info("Mean iou:\t" + f"{iou.nanmean():.2f}")
        logger.info("Mean prec:\t" + f"{precision.nanmean():.2f}")
        logger.info("Mean recal:\t" + f"{recall.nanmean():.2f}")
    if writer is not None:
        writer.add_scalar(f'Validation miou', iou.nanmean().item(), step)
        writer.add_scalar(f'Validation average precision', precision.nanmean().item(), step)
        writer.add_scalar(f'Validation average recall', recall.nanmean().item(), step)
    return iou.nanmean(), precision.nanmean(), recall.nanmean()
