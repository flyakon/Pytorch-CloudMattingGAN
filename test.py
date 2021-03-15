import torch
import argparse
import numpy as np
import mmcv
from CloudMatting.models.builder import builder_models
from CloudMatting.utils import utils

parse=argparse.ArgumentParser()
# parse.add_argument('--config_file',
#             default=r'configs/vr_resnet50_inapinting_agr_cfg.py',type=str)
parse.add_argument('--config_file',default=r'configs/cloud_LSGAN_resnet50_cfg.py',type=str)
parse.add_argument('--data_path',default=None,type=str)
parse.add_argument('--data_format',default=None,type=str)
parse.add_argument('--result_path',default=None,type=str)
parse.add_argument('--checkpoints_path',default=None,type=str)
parse.add_argument('--log_path',default=None,type=str)

if __name__=='__main__':
    args = parse.parse_args()
    print(args)
    cfg = mmcv.Config.fromfile(args.config_file)

    models=builder_models(**cfg['config'])

    run_args={}

    models.run_test_interface(checkpoint_path=args.checkpoints_path,
                                  log_path=args.log_path,
                                data_path=args.data_path,
                                data_format=args.data_format,
                                result_path=args.result_path)