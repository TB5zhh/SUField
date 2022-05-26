import logging
import os
import sys


def setup_logger(rank, filename=None):
    format = f"[{os.uname()[1].split('.')[0]}]" + f'[%(asctime)s][%(levelname)s][{rank}][%(filename)s@%(lineno)d] %(message)s'
    datefmt = '%Y/%m/%d %H:%M:%S'
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARN)
    fh = logging.FileHandler(filename + f'{rank}.log') if filename is not None else logging.NullHandler()
    fh.setLevel(logging.INFO)
    logging.basicConfig(format=format, datefmt=datefmt, handlers=[ch, fh], level=logging.NOTSET)
