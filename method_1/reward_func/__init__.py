import os

import torch
from omegaconf import OmegaConf


def get_trained_model(load_from, default_cfg=None, return_optimizer=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load cfg
    cfg_file_path = os.path.join(load_from, 'cfg.yaml')
    loaded_cfg = OmegaConf.load(cfg_file_path)
    # merge with default cfg if default cfg is not None
    if default_cfg is not None:
        OmegaConf.update(default_cfg, 'reward_func',
                         loaded_cfg.reward_func,
                         force_add=True)
        loaded_cfg = default_cfg

    # merge with the corresponding attr_func config
    cfg_file_path = os.path.join(loaded_cfg.reward_func.attr_func_path, 'cfg.yaml')
    attr_func_cfg = OmegaConf.load(cfg_file_path)
    OmegaConf.update(loaded_cfg, 'attr_func',
                     attr_func_cfg.attr_func,
                     force_add=True)
    # override the language rep option
    if 'use_language_attr' in attr_func_cfg:
        loaded_cfg.use_language_attr = attr_func_cfg.use_language_attr

    # load attr_info
    domain_description = OmegaConf.load(loaded_cfg.domain_description_file)
    OmegaConf.update(loaded_cfg, 'attr_info',
                     domain_description.attr_info,
                     force_add=True)

    # load model
    model_params = torch.load(os.path.join(load_from, 'model.tar'), map_location=device)
    from method_1.reward_func.model import Reward_Func
    reward_func = Reward_Func(loaded_cfg).to(device)
    reward_func.load_state_dict(model_params)

    # load optimizer
    optimizer = None
    if return_optimizer:
        raise NotImplementedError
    return reward_func, optimizer, loaded_cfg
