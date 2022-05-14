import logging as L
from configparser import SectionProxy as Sec
import copy
#from MinkowskiEngine import SparseTensor
import MinkowskiEngine as ME
import numpy as np
import pointnet2._ext as p2
import torch
from torch import nn
from torch.serialization import default_restore_location

from sufield.lib.distributed_utils import all_gather_list, get_rank, get_world_size
from sufield.lib.solvers import initialize_optimizer, initialize_scheduler
from sufield.lib.test import test
from sufield.lib.utils import AverageMeter, Timer, checkpoint
from IPython import embed
import wandb


def validate(model, val_data_loader, curr_iter, config, transform_data_fn):
    v_loss, v_score, v_mAP, v_mIoU = test(model, val_data_loader, config, transform_data_fn)
    # TODO log
    return v_mIoU


def load_state(model, state):
    if get_world_size() > 1:
        _model = model.module
    else:
        _model = model
    _model.load_state_dict(state)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):

    def __init__(self, coef=3.9e-3) -> None:
        super().__init__()
        self.coef = coef

    def forward(self, x, y):
        x_norm = (x - x.mean(dim=0)) / x.std(dim=0)
        y_norm = (y - y.mean(dim=0)) / y.std(dim=0)
        cov = x_norm.t() @ y_norm

        x_l2 = x.pow(2).sum(0, keepdim=True).sqrt()
        y_l2 = y.pow(2).sum(0, keepdim=True).sqrt()
        cov = cov / (x_l2.t() @ y_l2)

        ret = (cov - torch.eye(cov.shape[0], device=cov.device)).pow(2)
        coef = self.coef * torch.ones_like(cov, device=cov.device) + (1 - self.coef) * torch.eye(cov.shape[0], device=cov.device)
        ret = ret * coef

        loss = ret.sum()
        return loss


def train(model, train_dataloader, conf: Sec):
    device = f'cuda:{get_rank()}'
    distributed = get_world_size() > 1

    model.train()

    #################### Recorders #####################
    data_timer, iter_timer = Timer(), Timer()
    fw_timer, bw_timer, ddp_timer = Timer(), Timer(), Timer()
    step_timer = Timer()

    ####################################################

    # Train the network
    L.info(f"Start training on {get_world_size()} GPUs")
    L.info(f"Batch size: {conf.getint('TrainBatchSize') * get_world_size() *conf.getint('LossAccumulateIter')}")

    optimizer = initialize_optimizer(model.parameters(), conf)
    scheduler = initialize_scheduler(optimizer, conf)
    # criterion = nn.CrossEntropyLoss(ignore_index=conf.getint('IgnoreLabel'))
    criterion = BarlowTwinsLoss(conf.getfloat('BarlowTwinsCoef'))

    if conf.getboolean('Resume'):
        ckpt_path = f"{conf['CheckpointLoadPath']}/{conf['RunName']}/latest.pth"
        state = torch.load(ckpt_path)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

        loss_step, loss_average = state['loss_step'], state['loss_average']
        global_iter = state['global_iter']
        global_step = state['global_step']
    else:
        loss_step, loss_average = AverageMeter(), AverageMeter()
        global_iter = 0
        global_step = 0

    validate_flag = False
    data_iter = train_dataloader.__iter__()
    epoch_length = len(train_dataloader)

    step_timer.tic()
    while True:
        iter_timer.tic()
        if validate_flag:
            # Validate
            validate_flag = False
            # TODO validate
            continue

        global_iter += 1

        ######### Data Processing Start
        data_timer.tic()
        coords1, coords2, input1, input2, target1, target2 = data_iter.next()

        # TODO check this
        # For some networks, making the network invariant to even, odd coords is important. Random translation
        coords1[:, 1:] += (torch.rand(3) * 100).type_as(coords1)
        coords2[:, 1:] += (torch.rand(3) * 100).type_as(coords2)

        # Preprocess input
        if conf.getboolean('NormalizeColor'):
            input1[:, :3] = input1[:, :3] / 255. - 0.5
            input2[:, :3] = input2[:, :3] / 255. - 0.5
        tfield1 = ME.TensorField(coordinates=coords1.int().to(device),
                                 features=input1.to(device),
                                 quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        tfield2 = ME.TensorField(coordinates=coords1.int().to(device),
                                 features=input1.to(device),
                                 quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        sinput1 = tfield1.sparse()
        sinput2 = tfield2.sparse()

        data_timer.toc(False)
        ######### Data Processing End

        ######### Feed Forward Start
        fw_timer.tic()
        inputs1 = (sinput1,)
        inputs2 = (sinput2,)
        soutput1 = model(*inputs1)
        soutput2 = model(*inputs2)
        ofield1 = soutput1.slice(tfield1)
        ofield2 = soutput2.slice(tfield2)

        target1 = target1.long().to(device)
        target2 = target2.long().to(device)

        pindex1 = p2.furthest_point_sampling(ofield1.C[:, 1:].reshape((1, ofield1.C.shape[0], 3)).contiguous(), 1024).reshape(1024).long()
        pindex2 = p2.furthest_point_sampling(ofield2.C[:, 1:].reshape((1, ofield2.C.shape[0], 3)).contiguous(), 1024).reshape(1024).long()
        list1 = torch.index_select(ofield1.F, 0, pindex1)
        list2 = torch.index_select(ofield2.F, 0, pindex1)

        # loss_iter = criterion(list1, list2)
        loss_iter = criterion(list1, list2)
        loss_iter /= conf.getint('LossAccumulateIter')

        # TODO get prediction
        # TODO precision
        fw_timer.toc(False)
        ######### Feed Forward End

        ######### Loss Backward Start
        bw_timer.tic()
        loss_iter.backward()
        optimizer.zero_grad()
        bw_timer.toc(False)
        ######### Loss Backward End

        ######### Distributed Sync Start
        ddp_timer.tic()
        if distributed:
            loss_iter = np.mean(all_gather_list(loss_iter))
        loss_step.update(loss_iter.item())
        loss_average.update(loss_iter.item())

        ddp_timer.toc(False)
        ######### Distributed Sync End

        iter_timer.toc(False)

        if conf.getboolean('DoIterLog'):
            L.info(f"Iter #{global_iter}: " + f"LR: {scheduler.get_last_lr()[-1]:.3f}, " + f"Iter Loss: {loss_iter.item():.4f}, " + f"Iter Time: {iter_timer.diff:.2f}, " +
                   f"Data Time: {data_timer.diff:.2f}, " + f"Feed Forward Time: {fw_timer.diff:.2f}, " + f"Backward Time: {bw_timer.diff:.2f}, " +
                   f"DDP Time: {ddp_timer.diff:.2f}")

        if global_iter % conf.getint('LossAccumulateIter') == 0:
            global_step += 1
            optimizer.step()
            scheduler.step()
            step_timer.toc(False)
            if conf.getboolean('DoStepLog') and global_step % conf.getint('LoggingStep') == 0:
                L.info(f"\tStep #{global_step}: " + f"Step Loss: {loss_step.avg:.4f}, " + f"Avg Loss: {loss_average.avg:.4f}" + f"Step Time: {step_timer.diff:.2f}")
            if conf.getboolean('SaveCheckpoint') and global_step % conf.getint('CheckpointStep') == 0:
                state_dict = {
                    'state_dict': model.module.state_dict() if get_world_size() > 1 else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'global_step': global_step,
                    'global_iter': global_iter,
                    'epoch': global_iter // epoch_length,
                    'loss_step': loss_step,
                    'loss_average': loss_average,
                    'conf': conf,
                    'step_size': get_world_size() * conf.getint('LossAccumulateIter') * conf.getint('TrainBatchSize'),
                }
                checkpoint(state_dict, conf, 'latest')

            if conf.getboolean('DoValidate') and global_step % conf.getint('ValidateStep') == 0:
                validate_flag = True
            if conf.getboolean('DoEmptyCache') and conf.getint('EmptyCacheStep'):
                torch.cuda.empty_cache()

            if conf.getboolean('UseWandb') and get_rank() == 0:
                wandb.log({
                    'global_step': global_step,
                    'global_iter': global_iter,
                    'epoch': global_iter // epoch_length,
                    'loss_step': loss_step.avg,
                    'loss_average': loss_average.avg,
                    'learning_rate': scheduler.get_last_lr()[-1]
                })
            step_timer.tic()
            loss_step.reset()

        if conf['MaxIteration'] != '' and global_iter > conf.getint('MaxIteration'):
            L.info(f"Training stopped due to MaxIteration:{conf.getint('MaxIteration')} limit")
            break
        if conf['MaxEpoch'] != '' and global_iter > epoch_length * conf.getint('MaxEpoch'):
            L.info(f"Training stopped due to MaxEpoch:{conf.getint('MaxEpoch')} limit")
            break

    # Save the final model
    state_dict = {
        'state_dict': model.module.state_dict() if get_world_size() > 1 else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'global_iter': global_iter,
        'epoch': global_iter // epoch_length,
        'loss_step': loss_step,
        'loss_average': loss_average,
        'conf': conf,
        'step_size': get_world_size() * conf.getint('LossAccumulateIter') * conf.getint('TrainBatchSize'),
    }
    checkpoint(state_dict, conf, 'last')
