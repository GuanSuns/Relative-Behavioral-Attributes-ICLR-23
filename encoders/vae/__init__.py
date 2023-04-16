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
        OmegaConf.update(default_cfg, 'encoder_cfg',
                         loaded_cfg.encoder_cfg,
                         force_add=True)
        loaded_cfg = default_cfg

    # load model
    model_params = torch.load(os.path.join(load_from, 'model.tar'), map_location=device)
    from encoders.vae.model import Behavior_VAE
    behavior_vae = Behavior_VAE(loaded_cfg).to(device)
    behavior_vae.load_state_dict(model_params)

    # load optimizer
    optimizer = None
    if return_optimizer:
        raise NotImplementedError
    return behavior_vae, optimizer, loaded_cfg
