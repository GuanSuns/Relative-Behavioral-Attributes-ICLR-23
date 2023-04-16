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
    from data import get_attribute_data
    attr_training_data, attr_test_data = get_attribute_data(cfg, read_test_data=True)

    # determine whether to use language attr representation
    expr_name_prefix = ''
    if 'use_language_attr' in cfg and cfg.use_language_attr:
        expr_name_prefix = 'language-'

    logger = Experiment_Tracker(saved_dir='results',
                                expr_name=expr_name_prefix + f'attr-func-{cfg.dataset_name}',
                                project_name='relative-attribute-method-1', use_wandb=sys_args.use_wandb)
    cfg.attr_func.expr_id = logger.expr_name
    logger.redirect_output_to_logfile_as_well()
    logger.save_config(cfg)

    from method_1.attr_func.trainer import Attr_Func_Trainer
    trainer = Attr_Func_Trainer(cfg, attr_training_data,
                                eval_data=attr_test_data,
                                load_from=sys_args.load_from,
                                logger=logger)
    trainer.train()
    logger.finish()


if __name__ == '__main__':
    main()
