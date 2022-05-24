import torch
import logging

from pyparsing import col
from sufield.datasets.sampler import DistributedInfSampler, InfSampler
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel


from ..datasets import transforms as t
from ..datasets import get_transform
from ..datasets.dataset import ScanNetVoxelizedDataset
from ..datasets.transforms import cf_collate_fn_factory
from ..models.viewpoint_bottleneck import ViewpointBottleneck
from .distributed_launch import run_distributed
from .utils import get_args, get_rank, get_world_size, AverageMeter, Timer, setup_logger
from .visualize import dump_points_with_labels


def train(args):
    args = args['training']
    rank = get_rank()
    torch.cuda.set_device(rank)
    world_size = get_world_size()
    setup_logger(rank, 'test.log')
    
    device = f"cuda:{rank}"
    logger = logging.getLogger(__name__)
    logger.debug('Train func start')
    """
    Timers and Meters
    """
    data_timer, iter_timer = Timer(), Timer()
    fw_timer, bw_timer, ddp_timer = Timer(), Timer(), Timer()
    step_timer = Timer()
    """
    Dataset, Transforms and Dataloaders
    """
    transforms = t.Compose( get_transform(args['transforms']))

    dataset_args = args['dataset']
    dataset = ScanNetVoxelizedDataset(dataset_args['data_list'], dataset_args['label_list'], return_paths=True, transforms=transforms)

    dataloader = DataLoader(dataset,
                            batch_size=args['batch_size'],
                            num_workers=args['num_workers'],
                            collate_fn=cf_collate_fn_factory(args['limit_numpoints']),
                            sampler=DistributedInfSampler(dataset),
                            pin_memory=True)
    logger.debug('Dataset and dataloader init')
    """
    Models
    """
    model = ViewpointBottleneck(None, None, None)
    model.cuda()
    model = DistributedDataParallel(model, device_ids=[rank])
    logger.debug('Model init')
    """
    Optimizer and Scheduler
    """
    optimizer_args = args['optimizer']
    if optimizer_args['type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=optimizer_args['learning_rate'],
                              momentum=optimizer_args['SGD']['momentum'],
                              dampening=optimizer_args['SGD']['dampening'],
                              weight_decay=optimizer_args['SGD']['dampening'])
    else:
        raise NotImplementedError

    scheduler_args = args['scheduler']
    if scheduler_args['type'] == 'Polynomial':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: (1 - epoch / args['max_iter'])**scheduler_args['poly']['power'],
        )
    # """
    # TODO Resuming
    # """
    # """
    # Training starts
    # """
    logger.debug('Start loop')
    for step_idx, sample in enumerate(dataloader):
        loss = model(sample)
        logger.info(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    args = get_args('config.yaml')
    run_distributed(2, train, args)
