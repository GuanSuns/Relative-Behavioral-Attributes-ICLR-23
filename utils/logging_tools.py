from datetime import datetime
import os
import sys
import shutil
import pickle

import numpy as np
import imageio
import torch
import wandb
import json

from addict import Dict
from flatten_dict import flatten
from omegaconf import OmegaConf


class Wandb_Logger:
    def __init__(self, proj_name, run_name):
        wandb.init(project=proj_name, name=run_name)

    def log(self, log_dict, prefix='', step=None):
        log_info = log_dict
        if prefix != '':
            log_info = dict()
            for k in log_dict:
                log_info[prefix + '/' + k] = log_dict[k]
        if step is not None:
            wandb.log(log_info, step=step)
        else:
            wandb.log(log_info)


DEFAULT_PROJECT_NAME = 'default-project'        # default project name
DEFAULT_IS_USE_WANDB = False        # whether to use wandb for logging
DEFAULT_SAVE_TRAJ = False  # whether to save transitions/trajectories


class Experiment_Tracker:
    def __init__(self, saved_dir, expr_name, log_note=None, add_time_to_name=True,
                 project_name=DEFAULT_PROJECT_NAME,
                 use_wandb=DEFAULT_IS_USE_WANDB):
        # add timestamp to expr name
        full_expr_name = expr_name
        if add_time_to_name:
            now = datetime.now()
            d = now.strftime('%Y%m%d_%H%M%S')
            full_expr_name = expr_name + '_' + d
        self.expr_name = full_expr_name
        self.short_expr_name = expr_name
        self.project_name = project_name
        self.log_note = log_note
        self.use_wandb = use_wandb
        if use_wandb:
            self.init_wandb()

        # create expr dir
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        self.saved_dir = os.path.join(saved_dir, full_expr_name)
        os.mkdir(self.saved_dir)

        # create image saved dir
        self.img_dir = os.path.join(self.saved_dir, 'images')
        os.mkdir(self.img_dir)

        # create snapshot saved dir
        self.snapshot_dir = os.path.join(self.saved_dir, 'snapshots')
        os.mkdir(self.snapshot_dir)

        # trajectories holder
        self.trajectories = list()    # traj with transition information
        self.current_trajectory = list()

        # create log file
        self.logfile = open(os.path.join(self.saved_dir, 'log.txt'), 'a')

    def save_models(self, checkpoint, cfg=None, postfix=None, is_snapshot=True, prefix='model'):
        """
        Save current model
        :param checkpoint: the parameters of the models, see example in pytorch's documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        :param is_snapshot: whether saved in the snapshot directory
        :param cfg: the config used for training the model
        :param prefix: the prefix of the file name
        :param postfix: the postfix of the file name (can be episode number, frame number and so on)
        """
        saved_dir = self.snapshot_dir if is_snapshot else self.saved_dir
        if postfix is not None:
            dir_name = get_unique_folder_name(saved_dir, prefix + '_' + postfix)
        else:
            dir_name = get_unique_folder_name(saved_dir, prefix + '_' + self.expr_name)
        saved_dir = os.path.join(saved_dir, dir_name)
        os.makedirs(saved_dir, exist_ok=True)
        # save model
        if isinstance(checkpoint, dict):
            for key in checkpoint:
                torch.save(checkpoint[key], os.path.join(saved_dir, f'{key}.tar'))
        else:
            torch.save(checkpoint, os.path.join(saved_dir, 'model.tar'))
        # save config
        if cfg is not None:
            OmegaConf.save(config=cfg, f=os.path.join(saved_dir, 'cfg.yaml'))

    def redirect_output_to_logfile_as_well(self):
        class Logger(object):
            def __init__(self, logfile):
                self.stdout = sys.stdout
                self.logfile = logfile

            def write(self, message):
                self.stdout.write(message)
                self.logfile.write(message)

            def flush(self):
                # this flush method is needed for python 3 compatibility.
                # this handles the flush command by doing nothing.
                # you might want to specify some extra behavior here.
                pass
        sys.stdout = Logger(self.logfile)
        sys.stderr = sys.stdout

    def init_wandb(self):
        if self.use_wandb:
            if self.log_note is None:
                wandb.init(project=self.project_name, name=self.expr_name)
            else:
                notes = f'{self.expr_name} | {self.log_note}'
                wandb.init(project=self.project_name, name=self.short_expr_name, notes=notes)

    def save_config(self, config):
        OmegaConf.save(config=config, f=os.path.join(self.saved_dir, 'expr_config.yaml'))
        if self.use_wandb:
            # flatten the config first
            flat_config = flatten(config, reducer='path')
            if self.use_wandb:
                wandb.config.update(flat_config)

    @staticmethod
    def log_wandb(log_dict, step=None):
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)

    def log(self, log_dict, step=None):
        if self.use_wandb:
            self.log_wandb(log_dict, step=step)

    def finish(self):
        if self.use_wandb:
            wandb.finish()


def get_unique_fname(file_dir, fname_base):
    name, extension = os.path.splitext(fname_base)
    post_fix = 0
    while True:
        fname = name + '_' + str(post_fix) + extension
        if not os.path.exists(os.path.join(file_dir, fname)):
            return fname
        post_fix += 1


def get_unique_folder_name(file_dir, fname_base):
    post_fix = 0
    while True:
        fname = fname_base + '_' + str(post_fix)
        if not os.path.exists(os.path.join(file_dir, fname)):
            return fname
        post_fix += 1


class Loss_Info(object):
    def __init__(self):
        self.loss_info = Dict()
        self.reset()

    def reset(self):
        for item in self.loss_info:
            self.loss_info[item] = list()

    def update(self, new_loss_info):
        for item in new_loss_info:
            if item not in self.loss_info:
                self.loss_info[item] = list()
            self.loss_info[item].append(new_loss_info[item])

    def avg(self):
        avg_loss_info = Dict()
        for item in self.loss_info:
            avg_loss_info[item] = np.mean(self.loss_info[item]) if len(self.loss_info[item]) > 0 else np.nan
        return avg_loss_info

    def get_log_dict(self):
        return Dict(self.avg())


