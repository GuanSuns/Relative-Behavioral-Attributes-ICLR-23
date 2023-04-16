import os

import torch
from omegaconf import OmegaConf


def get_trained_encoder(load_from, default_cfg=None, return_optimizer=False):
    # read encoder type from
    cfg_file_path = os.path.join(load_from, 'cfg.yaml')
    loaded_cfg = OmegaConf.load(cfg_file_path)
    encoder_type = loaded_cfg.encoder_cfg.encoder_type

    if encoder_type == 'vae':
        from encoders.vae import get_trained_model
        return get_trained_model(load_from, default_cfg=default_cfg,
                                 return_optimizer=return_optimizer)
    else:
        raise NotImplementedError

