import argparse
import logging
import os
import shutil
import sys

import git
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..datasets import get_transform
from ..datasets import transforms as t
from ..datasets.dataset import BundledDataset, ScanNetVoxelized
from ..datasets.sampler import DistributedInfSampler, InfSampler
from ..datasets.transforms import cf_collate_fn_factory
from ..models.viewpoint_bottleneck import ViewpointBottleneck
from .utils import (
    AverageMeter,
    Timer,
    current_timestr,
    get_args,
    get_rank,
    checkpoint,
    get_world_size,
    setup_logger,
    run_distributed,
)
from .visualize import get_correlated_map


def train(args):
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)
    setup_logger(rank, f"{args['output_dir']}/output.log")
    if args['resume'] is not None:
        state_dict = torch.load(args['resume'], map_location=f"cuda:{rank}")
    elif args['load'] is not None:
        state_dict = torch.load(args['load'], map_location=f"cuda:{rank}")
    training_args = args['training']

    logger = logging.getLogger(__name__)
    if rank == 0:
        os.makedirs(f"{args['output_dir']}/tensorboard", exist_ok=True)
        writer = SummaryWriter(f"{args['output_dir']}/tensorboard")
        repo = git.Repo(search_parent_directories=True)
        logger.info("Current commit id: " + repo.head.commit.hexsha)
        if repo.is_dirty():
            logger.warning("There are uncommited changes in this repo. It is recommended to commit changes before running experiments")
    logger.debug('Train func start')
    """
    Timers and Meters
    """
    data_timer, fw_timer, bw_timer = Timer(), Timer(), Timer()
    loss_avg, score_avg = AverageMeter(), AverageMeter()
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
    model = ViewpointBottleneck(training_args['arch'], training_args['mode'])
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
    elif args['load'] is not None:
        model.load_state_dict(state_dict['model'])

    # """
    # Training starts
    # """
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    logger.debug(f'Start loop: one epoch has {len(dataloader)} steps')
    scaler = torch.cuda.amp.GradScaler()
    for step_idx, sample in zip(range(start_step, training_args['max_iter']), dataloader):
        data_timer.toc()

        fw_timer.tic()
        with torch.cuda.amp.autocast():
            loss, ret = model(sample)
        fw_timer.toc()

        bw_timer.tic()
        assert loss.item() != float('nan')
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_avg.update(loss.item())
        bw_timer.toc()

        if training_args['logging_steps'] is not None and training_args['logging_steps'] > 0 and (step_idx + 1) % training_args['logging_steps'] == 0:
            logger.info(f"Step {step_idx:6d}/{training_args['max_iter']} "
                        f"Loss: {loss.item():.4f}({loss_avg.avg:.4f}) "
                        f"Data time: {data_timer.diff:.2f} "
                        f"Forward time: {fw_timer.diff:.2f} "
                        f"Backward time: {bw_timer.diff:.2f} ")
        if rank == 0:
            writer.add_scalar(f'Loss/{training_args["mode"]}', loss.item(), step_idx)
            if training_args['mode'] == 'SSRL':
                writer.add_image(f'Correlated Map', get_correlated_map(ret**0.1), dataformats='HWC', global_step=step_idx)

        if training_args['checkpoint_steps'] is not None and training_args['checkpoint_steps'] > 0 and (step_idx + 1) % training_args['checkpoint_steps'] == 0:
            checkpoint(args, model.module if world_size > 1 else model, optimizer, scheduler, step_idx, None)

        if training_args['validate_steps'] is not None and training_args['validate_steps'] and (step_idx + 1) % training_args['validate_steps'] == 0:
            raise NotImplementedError

        data_timer.tic()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    resume = parser.parse_args().resume
    cfg_path = parser.parse_args().config
    if resume is not None:
        state_dict = torch.load(resume, map_location='cpu')
        args = state_dict['args']
        args['resume'] = resume
    else:
        args = get_args(cfg_path)
        args['start_time'] = current_timestr()
        args['output_dir'] = f"{args['log_root_dir']}/{args['exp_name']}/{args['start_time']}"
        os.makedirs(args['output_dir'], exist_ok=True)
        shutil.copy(cfg_path, args['output_dir'] + '/config.yaml')
        args['resume'] = None
    run_distributed(args['training']['world_size'], train, args)
