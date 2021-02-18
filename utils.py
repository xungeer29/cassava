# @Author: yican
# @Date: 2020-06-14 16:19:48
# @Last Modified by:   yican
# @Last Modified time: 2020-06-30 10:11:22
# Standard libraries
import logging
import os
import random
from argparse import ArgumentParser
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

# Third party libraries
import cv2
import numpy as np
import pandas as pd
import torch

IMG_SHAPE = (600, 800, 3)
IMAGE_FOLDER = "data/train_images"
NPY_FOLDER = "/home/public_data_center/kaggle/plant_pathology_2020/npys"
LOG_FOLDER = "logs"


def mkdir(path: str):
    """Create directory.

     Create directory if it is not exist, else do nothing.

     Parameters
     ----------
     path: str
        Path of your directory.

     Examples
     --------
     mkdir("data/raw/train/")
     """
    try:
        if path is None:
            pass
        else:
            os.stat(path)
    except Exception:
        os.makedirs(path)


def seed_reproducer(seed=2020):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def init_hparams():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("-b", "--backbone", type=str, default="tf_efficientnet_b4_ns") # efficientnet-b1, se_resnet50  vit_base_patch16_384
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=8 * 1)
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=32 * 1)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--image_size", nargs="+", default=[512, 512]) # 320, 416 512
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument('--cache', action='store_true', help='load data to memory')
    
    parser.add_argument("--gpus", nargs="+", default=[0,])  # 输入1 2 3
    parser.add_argument("--precision", type=int, default=32) # 16 or 32
    parser.add_argument("--gradient_clip_val", type=float, default=1)
    parser.add_argument("--soft_labels_filename", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    parser.add_argument("--version", type=str, default="debug")
    parser.add_argument('--ft', action='store_true', help='fine tuning')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER', help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Optimizer momentum (default: 0.9)') # sgd
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--freeze', action='store_true', help='freeze layers')

    # Learning rate schedule parameters
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 2)')
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER', choices=['step', 'cosine'], help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 0.01)') # 1e-4
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR', help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=8, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument("--T_max", type=int, default=10)
    parser.add_argument('--warmup', action='store_true', help='use warmup lr scheduler')

    parser.add_argument('--onehot', action='store_true', help='use onehot label')
    parser.add_argument("--smooth", type=float, default=0.7, help='label smooth value, 1 is not using label_smooth.') # 0.7

    # data augmentations
    parser.add_argument("--phflip", type=float, default=0.5, help='the prob of HFlip.')
    parser.add_argument("--pvflip", type=float, default=0.5, help='the prob of VFlip.')
    parser.add_argument("--ptranspose", type=float, default=0.5, help='the prob of transpose.')
    parser.add_argument('-pssr', "--pshiftscalerotate", type=float, default=0.5, help='the prob of ShiftScaleRotate.')
    parser.add_argument('-phue', "--phuesaturationvalue", type=float, default=0.5, help='the prob of HueSaturationValue.')
    parser.add_argument('-pbright', "--prandombrightnesscontrast", type=float, default=0.5, help='the prob of RandomBrightnessContrast.')
    parser.add_argument("--pcoarsedropout", type=float, default=0.5, help='the prob of RandomBrightnessContrast.')
    parser.add_argument("--pcutout", type=float, default=0.5, help='the prob of RandomBrightnessContrast.')
    # -------------
    parser.add_argument("--mixup", type=float, default=0., help='the prob of mixup, mixup=0 will close mixup.')
    parser.add_argument("--mixup_beta", type=float, default=0.4, help='beta in mixup.')
    parser.add_argument("--ricap", type=float, default=0, help='the prob of RICAP, ricap=0 will close RICAP.')
    parser.add_argument("--ricap_beta", type=float, default=0.2, help='beta in ricap.')
    parser.add_argument("--cutmix", type=float, default=0., help='the prob of cutmix, cutmix=0 will close cutmix.')
    parser.add_argument("--cutmix_beta", type=float, default=1.0, help='beta in cutmix.')
    parser.add_argument("--fmix", type=float, default=0, help='the prob of fmix, fmix=0 will close fmix.')
    parser.add_argument("--fmix_beta", type=float, default=1.0, help='beta in fmix.')
    parser.add_argument("--fmix_delta", type=float, default=3, help='decay power (delta) in fmix.') # 
    parser.add_argument("--snapmix", type=float, default=0., help='the prob of snapmix, snapmix=0 will close snapmix.')
    parser.add_argument("--snapmix_beta", type=float, default=5, help='beta in snapmix.')

    parser.add_argument('--use2019', action='store_true', help='use 2019 dataset')
    parser.add_argument("--sampler", type=str, default="common", choices=['common', 'balance'], help='data sampler method')

    try:
        hparams = parser.parse_args()
    except:
        hparams = parser.parse_args([])
    print("GPU:", type(hparams.gpus), hparams.gpus)
    if len(hparams.gpus) == 1:
        hparams.gpus = [int(hparams.gpus[0])]
    else:
        hparams.gpus = [int(gpu) for gpu in hparams.gpus]

    # hparams.image_size = [int(size) for size in hparams.image_size]
    # num = max([int(f.split('_')[-1]) for f in os.listdir(hparams.log_dir)])
    # hparams.version = f'version_{num+1}'
    
    return hparams


def load_data(logger, frac=1):
    data, test_data = pd.read_csv("data/train.csv"), pd.read_csv("data/sample_submission.csv")
    # Do fast experiment
    if frac < 1:
        logger.info(f"use frac : {frac}")
        data = data.sample(frac=frac).reset_index(drop=True)
        # test_data = test_data.sample(frac=frac).reset_index(drop=True)
    return data, test_data


def init_logger(log_name, log_dir=None):
    """日志模块
    Reference: https://juejin.im/post/5bc2bd3a5188255c94465d31
    日志器初始化
    日志模块功能:
        1. 日志同时打印到到屏幕和文件
        2. 默认保留近一周的日志文件
    日志等级:
        NOTSET（0）、DEBUG（10）、INFO（20）、WARNING（30）、ERROR（40）、CRITICAL（50）
    如果设定等级为10, 则只会打印10以上的信息

    Parameters
    ----------
    log_name : str
        日志文件名
    log_dir : str
        日志保存的目录

    Returns
    -------
    RootLogger
        Python日志实例
    """

    mkdir(log_dir)

    # 若多处定义Logger，根据log_name确保日志器的唯一性
    if log_name not in Logger.manager.loggerDict:
        logging.root.handlers.clear()
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        # 定义日志信息格式
        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s] %(filename)s[%(lineno)4s] : %(levelname)s  %(message)s"
        formatter = logging.Formatter(format_str, datefmt)

        # 日志等级INFO以上输出到屏幕
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            # 日志等级INFO以上输出到{log_name}.log文件
            file_info_handler = TimedRotatingFileHandler(
                filename=os.path.join(log_dir, "%s.log" % log_name), when="D", backupCount=7
            )
            file_info_handler.setFormatter(formatter)
            file_info_handler.setLevel(logging.INFO)
            logger.addHandler(file_info_handler)

    logger = logging.getLogger(log_name)

    return logger


def read_image(image_path):
    """ 读取图像数据，并转换为RGB格式
        32.2 ms ± 2.34 ms -> self
        48.7 ms ± 2.24 ms -> plt.imread(image_path)
    """
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
