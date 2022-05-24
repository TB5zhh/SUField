import logging
import os
import sys


def setup_logger(rank, filename=None):
    format = f"[{os.uname()[1].split('.')[0]}]" + '[%(asctime)s][%(levelname)s][%(module)s][%(filename)s@%(lineno)d] %(message)s'
    datefmt = '%Y/%m/%d %H:%M:%S'
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        fh = logging.FileHandler(filename) if filename is not None else logging.NullHandler()
        fh.setLevel(logging.INFO)
        logging.basicConfig(
            format=format,
            datefmt=datefmt,
            handlers=[ch, fh])
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARN)
        fh = logging.FileHandler(filename) if filename is not None else logging.NullHandler()
        fh.setLevel(logging.WARN)
        logging.basicConfig(
            format=format,
            datefmt=datefmt,
            handlers=[ch])
