import configparser
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import random
import logging
from sufield.utils import get_rank, get_world_size, setup_logging


def main():
    conf = configparser.ConfigParser()
    conf['RUN_TIME'] = {}
    num_devices = torch.cuda.device_count()
    conf['RUN_TIME']['world_size'] = str(num_devices)

    if num_devices > 1:
        port = random.randint(10000, 20000)
        conf['RUN_TIME']['dist_init_method'] = f'tcp://localhost:{port}'
        mp.spawn(
            fn=distributed_main,
            args=(conf,),
            nprocs=num_devices,
        )
    else:
        main_worker(conf)


def distributed_main(i, conf):
    main_worker(conf, rank=i, world_size=conf['RUN_TIME'].getint('world_size'))


def main_worker(conf, rank=0, world_size=1):

    ###### Initialize distributed ######
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method=conf['RUN_TIME']['dist_init_method'], world_size=world_size, rank=rank)

    setup_logging(conf)

    # TODO print config

    