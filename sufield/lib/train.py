import argparse
import os
import shutil
import sys
from xml.etree.ElementInclude import default_loader
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
from .visualize import dump_correlated_map, dump_points_with_labels


def train(args):
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)
    setup_logger(rank, f"{args['output_dir']}/output.log")
    if args['resume'] is not None:
        state_dict = torch.load(args['resume'], map_location=f"cuda:{rank}")
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
                            sampler=DistributedInfSampler(dataset) if world_size > 1 else InfSampler(dataset),
                            pin_memory=True)
    logger.debug('Dataset and dataloader init')
    """
    Models
    """
    model = ViewpointBottleneck(None, None, None)
    model.cuda()
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
    start_step = 0
    if args['resume'] is not None:
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        start_step = state_dict['step'] + 1

    # """
    # Training starts
    # """
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    logger.debug(f'Start loop: one epoch has {len(dataloader)} steps')
    scaler = torch.cuda.amp.GradScaler()
    for step_idx, sample in zip(range(start_step, training_args['max_iter']), dataloader):
        with torch.cuda.amp.autocast():
            loss, ret = model(sample)
        assert loss.item() != float('nan')
        logger.info(f"Step {step_idx:6d}/{training_args['max_iter']} : {loss.item():.4f}")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if training_args['checkpoint_steps'] is not None and training_args['checkpoint_steps'] > 0 and (step_idx + 1) % training_args['checkpoint_steps'] == 0:
            checkpoint(args, model.module if world_size > 1 else model, optimizer, scheduler, step_idx, None)

        if training_args['validate_steps'] is not None and training_args['validate_steps'] and (step_idx + 1) % training_args['validate_steps'] == 0:
            if rank == 0:
                dump_correlated_map(ret, f'test-{step_idx}.png')
                dump_correlated_map(ret / ret.max(), f'test-relative-{step_idx}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    resume = parser.parse_args().resume
    if resume is not None:
        state_dict = torch.load(resume, map_location='cpu')
        args = state_dict['args']
        args['resume'] = resume
    else:
        args = get_args('config.yaml')
        args['start_time'] = current_timestr()
        args['output_dir'] = f"{args['log_root_dir']}/{args['exp_name']}/{args['start_time']}"
        os.makedirs(args['output_dir'], exist_ok=True)
        shutil.copy('config.yaml', args['output_dir'] + '/config.yaml')
        args['resume'] = None
    run_distributed(args['training']['world_size'], train, args)
