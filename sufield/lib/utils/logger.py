import logging
import os
import sys


def setup_logger(rank, filename=None, ch_level=logging.WARN, fh_level=logging.INFO):
    format = f"[{os.uname()[1].split('.')[0]}]" + f'[%(asctime)s][%(levelname)s][{rank}][%(filename)s@%(lineno)d] %(message)s'
    datefmt = '%Y/%m/%d %H:%M:%S'
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(ch_level)
    handlers = [ch]
    if filename is not None:
        fh = logging.FileHandler(filename + f'{rank}.log') if filename is not None else logging.NullHandler()
        fh.setLevel(fh_level)
        handlers.append(fh)
    logging.basicConfig(format=format, datefmt=datefmt, handlers=handlers, level=logging.NOTSET)
