import logging
from pyparsing import col

from torch import optim
from torch.utils.data import DataLoader

from sufield.models.viewpoint_bottleneck import ViewpointBottleneck

from .datasets import get_transform
from .datasets.dataset import ScanNetVoxelizedDataset
from .datasets.transforms import cf_collate_fn_factory
from .utils.distributed import get_rank, get_world_size
from .utils.meters import AverageMeter, Timer


def train(args):
    args = args['training']
    rank = get_rank()
    world_size = get_world_size()
    device = f"cuda:{rank}"
    logger = logging.getLogger(__name__)
    """
    Timers and Meters
    """
    data_timer, iter_timer = Timer(), Timer()
    fw_timer, bw_timer, ddp_timer = Timer(), Timer(), Timer()
    step_timer = Timer()
    """
    Dataset, Transforms and Dataloaders
    """
    transforms = get_transform(args['transforms'])

    dataset_args = args['dataset']
    dataset = ScanNetVoxelizedDataset(dataset_args['data_list'],
                                      dataset_args['label_list'],
                                      return_paths=True,
                                      transforms=transforms)

    dataloader = DataLoader(dataset,
                            batch_size=args['batch_size'],
                            num_workers=args['num_worker'],
                            collate_fn=cf_collate_fn_factory(args['limit_numpoints']))
    """
    Models
    """
    model = ViewpointBottleneck(None, None, None)
    model.cuda()

    
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
    """
    TODO Resuming
    """
    """
    Training starts  
    """
    while True:
        iter_timer.tic()
        # coords, feats, labels, *_ =
        
