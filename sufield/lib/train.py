import argparse
import logging
import os
import shutil

import git
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..datasets import get_transform
from ..datasets import transforms as t
from ..datasets.dataset import (BundledDataset, LimitedTrainValSplit,
                                ScanNetVoxelized, TrainValSplit)
from ..datasets.sampler import DistributedInfSampler, InfSampler
from ..datasets.transforms import cf_collate_fn_factory
from ..models.viewpoint_bottleneck import ViewpointBottleneck
from .utils import (AverageMeter, Timer, checkpoint, current_timestr,
                    deterministic, get_args, get_correlated_map, get_rank,
                    get_world_size, run_distributed, setup_logger)
from .validate import validate_pass


def train(args):
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)

    setup_logger(rank, f"{args['output_dir']}/output.log")
    if args['resume'] is not None:
        state_dict = torch.load(args['resume'], map_location=f"cuda:{rank}")
    elif args['load'] is not None:
        state_dict = torch.load(args['load'], map_location=f"cuda:{rank}")

    deterministic(args['seed'])
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
    train_transforms = t.Compose(get_transform(args['train']['transforms']))
    validate_transforms = t.Compose(get_transform(args['validate']['transforms']))

    if args['train']['mode'] == 'SSRL':
        train_dataset = ScanNetVoxelized(
            BundledDataset,
            bundle_path=f'{args["dataset"]["bundle_dir"]}/train.npy',
            transforms=train_transforms,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args['train']['batch_size'],
            num_workers=args['train']['num_workers'],
            collate_fn=cf_collate_fn_factory(args['train']['limit_numpoints']),
            sampler=DistributedInfSampler(train_dataset) if world_size > 1 else InfSampler(train_dataset),
            pin_memory=True,
        )
    elif args['train']['mode'] == 'Finetune':
        train_dataset = LimitedTrainValSplit(
            ScanNetVoxelized,
            BundledDataset,
            bundle_path=f'{args["dataset"]["bundle_dir"]}/train.npy',
            transforms=train_transforms,
            annotate_idx_dir=args["dataset"]["annotate_index_dir"],
            split_file_dir=args["dataset"]["split_file_dir"],
            split='train',
            size=args["train"]["annotation_count"],
            map_idx=3,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args['train']['batch_size'],
            num_workers=args['train']['num_workers'],
            collate_fn=cf_collate_fn_factory(args['train']['limit_numpoints']),
            sampler=DistributedInfSampler(train_dataset, shuffle=False) if world_size > 1 else InfSampler(train_dataset),
            pin_memory=True,
        )
        val_dataset = TrainValSplit(
            ScanNetVoxelized,
            BundledDataset,
            bundle_path=f'{args["dataset"]["bundle_dir"]}/train.npy',
            transforms=validate_transforms,
            split_file_dir=args["dataset"]["split_file_dir"],
            split='val',
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args['validate']['batch_size'],
            num_workers=args['validate']['num_workers'],
            collate_fn=cf_collate_fn_factory(args['validate']['limit_numpoints']),
            shuffle=False,
            pin_memory=True,
        )

    logger.debug('Dataset and dataloader init')
    """
    Models
    """
    model = ViewpointBottleneck(args['train']['arch'], args['train']['mode'])
    model.cuda()
    logger.debug('Model init')
    """
    Optimizer and Scheduler and AMP Scaler
    """
    if args['train']['optimizer']['type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args['train']['optimizer']['learning_rate'],
                              momentum=args['train']['optimizer']['SGD']['momentum'],
                              dampening=args['train']['optimizer']['SGD']['dampening'],
                              weight_decay=args['train']['optimizer']['SGD']['dampening'])
    else:
        raise NotImplementedError

    if args['train']['scheduler']['type'] == 'Polynomial':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: (1 - epoch / args['train']['max_iter'])**args['train']['scheduler']['poly']['power'],
        )
    scaler = torch.cuda.amp.GradScaler()
    """
    Resuming from checkpoints and DDP
    """
    start_step = 0
    if args['resume'] is not None:
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        scaler.load_state_dict(state_dict['scaler'])
        start_step = state_dict['step'] + 1
    elif args['load'] is not None:
        model.load_state_dict(state_dict['model'])
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    """
    Training starts
    """
    logger.info(f'Start loop: one epoch has {len(train_dataloader)} steps')
    for step_idx, sample in zip(range(start_step, args['train']['max_iter']), train_dataloader):
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

        if args['train']['logging_steps'] is not None and args['train']['logging_steps'] > 0 and (step_idx + 1) % args['train']['logging_steps'] == 0:
            logger.info(f"Step {step_idx:6d}/{args['train']['max_iter']} "
                        f"Loss: {loss.item():.4f}({loss_avg.avg:.4f}) "
                        f"Data time: {data_timer.diff:.2f} "
                        f"Forward time: {fw_timer.diff:.2f} "
                        f"Backward time: {bw_timer.diff:.2f} ")
        if rank == 0:
            writer.add_scalar(f'Loss/{args["train"]["mode"]}', loss.item(), step_idx)
            if args['train']['mode'] == 'SSRL':
                writer.add_image(f'Correlated Map', get_correlated_map(ret**0.1), dataformats='HWC', global_step=step_idx)

        if args['train']['checkpoint_steps'] is not None and args['train']['checkpoint_steps'] > 0 and (step_idx + 1) % args['train']['checkpoint_steps'] == 0:
            checkpoint(args, model.module if world_size > 1 else model, optimizer, scheduler, step_idx, None, scaler)

        if args['train']['validate_steps'] is not None and args['train']['validate_steps'] and (step_idx + 1) % args['train']['validate_steps'] == 0:
            validate_pass(model, val_dataloader, writer if rank == 0 else None, step_idx, logging=True)

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
    run_distributed(args['train']['world_size'], train, args)
