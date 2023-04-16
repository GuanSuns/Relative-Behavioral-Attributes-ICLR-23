import argparse
import os

import numpy as np
from addict import Dict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm


def main():
    sys_args = Dict()
    task_name = 'walker_step'
    sys_args.cfg_path = f'config/method_1/{task_name}_env.yaml'
    sys_args.attr_func_path = f'trained_models/{task_name}/method_1/attr_func/'

    # read config
    cfg = OmegaConf.load(sys_args.cfg_path)

    from method_1.attr_func import get_trained_model
    attr_func, _, cfg = get_trained_model(sys_args.attr_func_path, default_cfg=cfg, return_optimizer=False)

    # read dataset
    from data import get_attribute_data
    attr_training_data, attr_test_data = get_attribute_data(cfg, read_test_data=True)

    from method_1.attr_func.eval import eval_model
    from method_1.attr_func.eval import compute_attr_stats
    eval_loss_func = nn.CrossEntropyLoss(reduction='mean')
    eval_pred_thresholds = (0.55, 0.95)

    print('#' * 20)
    print('Training data')
    eval_model(attr_training_data, attr_func, eval_loss_func, cfg, eval_pred_thresholds, verbose=True)
    compute_attr_stats(attr_training_data, attr_func, verbose=True)
    print('#' * 20)
    print('Test data')
    eval_model(attr_test_data, attr_func, eval_loss_func, cfg, eval_pred_thresholds, verbose=True)
    compute_attr_stats(attr_test_data, attr_func, verbose=True)


if __name__ == '__main__':
    main()
