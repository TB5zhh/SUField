import configparser
import MinkowskiEngine as ME
import logging as L
import random
from configparser import SectionProxy as Sec
from IPython.terminal.embed import embed

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.serialization import default_restore_location

from sufield.lib.data.dataset import initialize_data_loader
from sufield.lib.data.datasets import load_dataset
from sufield.lib.train import train
from sufield.models import load_model
from sufield.utils import set_seeds, setup_logging
import wandb


def main():
    conf = configparser.ConfigParser()
    conf.read('conf.ini')
    print(f"Using section {conf['DEFAULT']['Section']} in configuration file")
    conf = conf[conf['DEFAULT']['Section']]
    num_devices = torch.cuda.device_count()
    conf['RT_world_size'] = str(num_devices)

    if num_devices > 1:
        port = random.randint(10000, 20000)
        conf['RT_dist_init_method'] = f'tcp://localhost:{port}'
        mp.spawn(
            fn=distributed_main,
            args=(conf,),
            nprocs=num_devices,
        )
    else:
        main_worker(conf)


def distributed_main(i, conf):
    main_worker(conf, rank=i, world_size=conf.getint('RT_world_size'))


def main_worker(conf: Sec, rank=0, world_size=1):
    conf['RT_rank'] = str(rank)
    setup_logging(conf)
    # TODO print config
    set_seeds(conf.getint('Seed'))
    assert torch.cuda.is_available(), "No GPU found or no CUDA found"

    ###### Initialize distributed ######
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method=conf['RT_dist_init_method'], world_size=world_size, rank=rank)

    DatasetClass = load_dataset(conf['Dataset'])

    ###### Initialize dataloader ######
    L.info('Initializing Dataloader...')
    if conf['Action'] == 'train':
        train_dataloader = initialize_data_loader(
            DatasetClass,
            conf,
            split=conf['TrainSplit'],
            num_workers=conf.getint('NumWorkers'),
            augment_data=True,
            shuffle=True,
            repeat=True,  # TODO check this
            batch_size=conf.getint('TrainBatchSize'),
            limit_numpoints=conf.getboolean('LimitNumPointsInTrainBatch'),
        )
        num_in_channel = 3
        num_labels = train_dataloader.dataset.NUM_LABELS
    elif conf['Action'] == 'test':
        raise NotImplementedError
    else:
        raise Exception(f'Unknown action: {conf["Action"]}')

    ###### Initialize model ######
    L.info(f'Initializing model ===> {conf["Model"]} <===...')
    NetClass = load_model(conf['Model'])
    model = NetClass(num_in_channel, num_labels, conf)

    # TODO log count of trainable parameters

    if conf['CheckpointLoadPath'] != '':
        ckpt = torch.load(conf['CheckpointLoadPath'], map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model.load_state_dict(ckpt['state_dict'])

    model = model.cuda(rank)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            broadcast_buffers=False,  # TODO
            bucket_cap_mb=25  # TODO
        )
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    if rank == 0 and conf.getboolean('UseWandb'):
        wandb.init(project="SUField", entity="tb5zhh")
        wandb.config = dict(conf)
        wandb.run.name = conf['RunName']
        wandb.run.save()
        wandb.watch(model if world_size == 1 else model.module)
    train(model, train_dataloader, conf)


if __name__ == '__main__':
    main()