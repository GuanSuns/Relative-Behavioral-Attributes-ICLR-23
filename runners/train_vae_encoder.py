import argparse
import os

from addict import Dict
from omegaconf import OmegaConf

from utils.logging_tools import Experiment_Tracker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', default=None, type=str)
    parser.add_argument('--load-from', default=None, type=str,
                        help="whether to continue from a pretrained model.")
    parser.add_argument("--use-wandb", dest='use_wandb', action="store_true",
                        help="use wandb for logging.")
    sys_args = Dict()
    args, unknown = parser.parse_known_args()
    for arg in vars(args):
        sys_args[arg] = getattr(args, arg)
    return sys_args


def main():
    sys_args = get_args()

    # read config
    cfg = OmegaConf.load(sys_args.cfg_path)

    # read dataset
    if cfg.is_video_data:
        from data.behavior_data import Video_Data
        training_data = Video_Data(data_path=os.path.join(cfg.dataset_dir, 'training_data.pickle'),
                                   frame_stack=cfg.frame_stack,
                                   image_size=cfg.image_size)
        test_data = Video_Data(data_path=os.path.join(cfg.dataset_dir, 'test_data.pickle'),
                               frame_stack=cfg.frame_stack,
                               image_size=cfg.image_size)
    else:
        raise NotImplementedError

    logger = Experiment_Tracker(saved_dir='results',
                                expr_name=f'behavior-encoder-{cfg.dataset_name}',
                                project_name='behavior-encoder', use_wandb=sys_args.use_wandb)
    cfg.encoder_cfg.expr_id = logger.expr_name
    cfg.encoder_cfg.encoder_type = 'vae'
    logger.redirect_output_to_logfile_as_well()
    logger.save_config(cfg)

    from encoders.vae.trainer import Behavior_VAE_Trainer
    trainer = Behavior_VAE_Trainer(cfg, training_data,
                                   eval_data=test_data,
                                   load_from=sys_args.load_from,
                                   logger=logger)
    trainer.train()
    logger.finish()


if __name__ == '__main__':
    main()
