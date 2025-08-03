import argparse
import logging
import sys


def default_logger(filename=None, mode='w'):
    formatter = logging.Formatter(fmt='[%(asctime)s][%(levelname)-5s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if filename:
        file_handler = logging.FileHandler(filename, mode=mode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    default_handler = logging.StreamHandler(stream=sys.stdout)
    default_handler.setFormatter(formatter)
    default_handler.setLevel(logging.INFO)
    logger.addHandler(default_handler)
    return logger

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)   

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def create_infinite_dataloader(loader):
    while True:
        yield from loader

def mask_fn(mode):
    if mode == 1:
        def wrapper(x):
            masked = x.clone()
            masked[:, :,  :, 14:] = 0
            masked[:, :, :14, :14] = 0
            return masked
    elif mode == 2:
        def wrapper(x):
            masked = x.clone()
            masked[:, :, :, 14:] = 0
            return masked
    elif mode == 3:
        def wrapper(x):
            masked = x.clone()
            masked[:, :, 14:, 14:] = 0
            return masked
    else:
        raise ValueError(f'unrecognized mode {mode}')
    return wrapper
