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
        prediction = model(sample)
        evaluator.update(prediction, sample[2])
    model.train(model_status)
    iou = evaluator.get_iou()
    precision = evaluator.get_precision()
    recall = evaluator.get_recall()
    if logging:
        logger.info("Validation result")
        logger.info("Per class iou:\t" + '\t'.join([f'{i:.2f}' for i in iou]))
        logger.info("Per class prec:\t" + '\t'.join([f'{i:.2f}' for i in precision]))
        logger.info("Per class recall:\t" + '\t'.join([f'{i:.2f}' for i in recall]))
    if writer is not None:
        writer.add_scalar(f'Validation miou', iou.nanmean().item(), step)
        writer.add_scalar(f'Validation average precision', precision.nanmean().item(), step)
        writer.add_scalar(f'Validation average recall', recall.nanmean().item(), step)
    return iou.nanmean(), precision.nanmean(), recall.nanmean()
