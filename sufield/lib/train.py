import os
import torch
import logging

from sufield.datasets.sampler import DistributedInfSampler, InfSampler
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from sufield.lib.utils.checkpoint import checkpoint


from ..datasets import transforms as t
from ..datasets import get_transform
from ..datasets.dataset import BundledDataset, ScanNetVoxelized
from ..datasets.transforms import cf_collate_fn_factory
from ..models.viewpoint_bottleneck import ViewpointBottleneck
from .distributed_launch import run_distributed
from .utils import (
    get_args,
    get_rank,
    get_world_size,
    AverageMeter,
    Timer,
    setup_logger,
    current_timestr,
)
from .visualize import dump_points_with_labels


def train(args):
    rank = get_rank()
    torch.cuda.set_device(rank)
    setup_logger(rank, f"{args['output_dir']}/output.log")
    training_args = args['training']

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
    transforms = t.Compose(get_transform(training_args['transforms']))

    dataset_args = training_args['dataset']
    dataset = ScanNetVoxelized(BundledDataset, bundle_path=f'{dataset_args["bundle_dir"]}/train.npy', transforms=transforms)

    dataloader = DataLoader(dataset,
                            batch_size=training_args['batch_size'],
                            num_workers=training_args['num_workers'],
                            collate_fn=cf_collate_fn_factory(training_args['limit_numpoints']),
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
    optimizer_args = training_args['optimizer']
    if optimizer_args['type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=optimizer_args['learning_rate'],
                              momentum=optimizer_args['SGD']['momentum'],
                              dampening=optimizer_args['SGD']['dampening'],
                              weight_decay=optimizer_args['SGD']['dampening'])
    else:
        raise NotImplementedError

    scheduler_args = training_args['scheduler']
    if scheduler_args['type'] == 'Polynomial':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: (1 - epoch / training_args['max_iter'])**scheduler_args['poly']['power'],
        )
    # """
    # TODO Resuming
    # """
    # """
    # Training starts
    # """
    logger.debug(f'Start loop: one epoch has {len(dataloader)} steps')
    for step_idx, sample in zip(range(training_args['max_iter']), dataloader):
        loss = model(sample)
        logger.info(f"Step {step_idx:6d}/{training_args['max_iter']} : {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if training_args['checkpoint_steps'] is not None and (step_idx + 1) % training_args['checkpoint_steps'] == 0:
            checkpoint(args, model, optimizer, scheduler, step_idx, None)


if __name__ == '__main__':
    args = get_args('config.yaml')
    args['start_time'] = current_timestr()
    args['output_dir'] = f"{args['log_root_dir']}/{args['exp_name']}/{args['start_time']}"
    os.makedirs(args['output_dir'], exist_ok=True)
    run_distributed(args['training']['world_size'], train, args)
