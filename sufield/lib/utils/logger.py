import logging
import os
import sys


def setup_logger(rank, filename=None):
    format = f"[{os.uname()[1].split('.')[0]}]" + f'[%(asctime)s][%(levelname)s][{rank}][%(filename)s@%(lineno)d] %(message)s'
    datefmt = '%Y/%m/%d %H:%M:%S'
    ch = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(filename) if filename is not None else logging.NullHandler()
    if rank == 0:
        logging.basicConfig(    
            format=format,
            datefmt=datefmt,
            handlers=[ch, fh], 
            level=logging.DEBUG)
    else:
        logging.basicConfig(
            format=format,
            datefmt=datefmt,
            handlers=[ch],
            level=logging.WARN)
