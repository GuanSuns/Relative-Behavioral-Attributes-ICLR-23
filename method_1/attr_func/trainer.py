import random

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from addict import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

import architectures.utils
from method_1.attr_func.eval import compute_attr_stats, eval_model
from utils import set_random_seed
from utils.logging_tools import Loss_Info


class Attr_Func_Trainer:
    def __init__(self, cfg, training_data, eval_data=None, load_from=None,
                 logger=None, log_freq=10, snapshot_freq=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        self.attr_func_cfg = self.cfg.attr_func
        self.logger = logger
        self.snapshot_freq = snapshot_freq
        self.eval_pred_thresholds = (0.51, 0.55)

        self.attr_rep_method = 'one-hot'
        if 'use_language_attr' in cfg and cfg.use_language_attr:
            self.attr_rep_method = 'language'

        if cfg.seed is None:
            cfg.seed = random.randint(1, 10000)
        set_random_seed(cfg.seed)

        if load_from is not None:
            from method_1.attr_func import get_trained_model
            self.attr_func, self.optimizer, self.cfg = get_trained_model(load_from, cfg, return_optimizer=True)
            self.attr_func_cfg = self.cfg.attr_func
        else:
            from method_1.attr_func.model import Attr_Func
            self.attr_func = Attr_Func(cfg).to(self.device)
            # init model
            self.attr_func.apply(architectures.utils.init_weights)
            self.optimizer = optim.Adam(self.attr_func.parameters(),
                                        lr=self.attr_func_cfg.lr, betas=(0.9, 0.999),
                                        weight_decay=self.attr_func_cfg.weight_decay)
        # parallel computing
        if torch.cuda.is_available():
            self.attr_func = torch.nn.DataParallel(self.attr_func)
        # cudnn.benchmark = True  # optimize for fixed input size
        print('[INFO] total params: %.2fM' % (sum(p.numel() for p in self.attr_func.parameters()) / 1000000.0))

        # init scheduler
        if self.attr_func_cfg.sche == "cosine":
            cosine_cfg = self.attr_func_cfg.cosine_sche
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.attr_func_cfg.n_epoch,
                                                                        eta_min=cosine_cfg.min_lr)
        elif self.attr_func_cfg.sche == "step":
            step_cfg = self.attr_func_cfg.step_sche
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=step_cfg.step_size,
                                                             gamma=step_cfg.gamma)
        elif self.attr_func_cfg.sche == "const":
            self.scheduler = None
        else:
            raise NotImplementedError

        # load training data
        self.training_data = training_data
        self.behavior_data = training_data.behavior_data
        self.train_loader = DataLoader(training_data,
                                       num_workers=4,
                                       batch_size=self.attr_func_cfg.batch_size,
                                       shuffle=True,
                                       drop_last=False,
                                       pin_memory=False)
        # load eval data
        self.eval_data = eval_data

        self.train_log_freq = max(1, len(self.train_loader) // log_freq)
        self.print_log_freq = max(1, len(self.train_loader) // 10)

        # define loss
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')
        self.epoch_loss_info = Loss_Info()
        self.batch_loss_info = Loss_Info()

    def save_model(self, epoch, is_snapshot=False):
        if self.logger is not None:
            try:
                model_state_dict = self.attr_func.module.state_dict()
            except AttributeError:
                model_state_dict = self.attr_func.state_dict()
            params = {
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
            }
            self.logger.save_models(checkpoint=params, cfg=self.cfg,
                                    is_snapshot=is_snapshot, postfix=str(epoch))

    def train(self):
        print(
            f'[INFO] start training for {self.attr_func_cfg.n_epoch} epoch, # of mini-batch: {len(self.train_loader)}')
        n_batch_update = 0
        for epoch in range(self.attr_func_cfg.n_epoch):
            if epoch and self.scheduler is not None:
                self.scheduler.step()

            self.epoch_loss_info.reset()

            for i, data in enumerate(self.train_loader):
                # torch_attr: torch.Size([batch_size, attr_dim])
                # torch_traj_0 : torch.Size([batch_size, trace_len, z_dim])
                # torch_mask_0: torch.Size([batch_size, trace_len])
                # torch_traj_1: torch.Size([batch_size, trace_len, z_dim])
                # torch_mask_1: torch.Size([batch_size, trace_len])
                # torch_label: torch.Size([batch_size, 2])
                torch_attr, torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, torch_label, sample_weights = data
                torch_attr = torch_attr.to(self.device)
                torch_traj_0 = torch_traj_0.to(self.device)
                torch_mask_0 = torch_mask_0.to(self.device)
                torch_traj_1 = torch_traj_1.to(self.device)
                torch_mask_1 = torch_mask_1.to(self.device)
                torch_label = torch_label.to(self.device)
                torch_sample_weights = sample_weights.to(self.device)
                # model update
                losses_info = self.train_batch(torch_attr, torch_traj_0, torch_mask_0,
                                               torch_traj_1, torch_mask_1,
                                               torch_label, torch_sample_weights, mode='train')
                n_batch_update += 1
                # save losses info
                self.epoch_loss_info.update(losses_info)
                self.batch_loss_info.update(losses_info)
                # logging: print info
                if i % self.print_log_freq == 0 and i:
                    lr = self.optimizer.param_groups[0]['lr']
                    cross_entropy_loss, avg_traj_rewards = losses_info['loss'], losses_info['avg_traj_reward']
                    accuracy = losses_info['accuracy']
                    print('[INFO] epoch %d, batch %d | loss: %.5f | acc: %.3f| avg traj reward: %.5f | lr: %.5f' % (
                            epoch, i, cross_entropy_loss, accuracy, avg_traj_rewards, lr))
                # logging: to wandb or other logging system
                if i % self.train_log_freq == 0 and i:
                    if self.logger is not None:
                        lr = self.optimizer.param_groups[0]['lr']
                        log_info_dict = self.batch_loss_info.get_log_dict()
                        log_info_dict.update({'update_step': n_batch_update, 'epoch': epoch, 'lr': lr})
                        self.logger.log(log_info_dict)
                    self.batch_loss_info.reset()
            # epoch logging
            log_info_dict = self.epoch_loss_info.get_log_dict()
            print(f'[INFO] epoch {epoch} - {log_info_dict}')
            if self.logger is not None:
                lr = self.optimizer.param_groups[0]['lr']
                log_info_dict.update({'update_step': n_batch_update, 'epoch': epoch, 'lr': lr})
                self.logger.log(log_info_dict)
            self.epoch_loss_info.reset()

            # snapshot
            if epoch and self.snapshot_freq is not None and epoch % self.snapshot_freq == 0:
                self.save_model(epoch, is_snapshot=True)
        # get eval loss
        if self.eval_data is not None:
            eval_stats = self.eval_epoch()
            eval_stats['curr_budget'] = self.training_data.n_trajectories
            eval_stats['curr_trajs'] = self.training_data.n_trajectories
            eval_stats['curr_labels'] = len(self.training_data)
            if self.logger is not None:
                self.logger.log(eval_stats)
            # print info
            eval_info_key = 'eval/all_attr/' + str(self.eval_pred_thresholds[0]) + '/all_acc'
            print(f'[INFO] eval acc: {eval_stats[eval_info_key]} | threshold: {self.eval_pred_thresholds[0]}.')
        self._compute_attr_stats()
        # save the trained model
        self.save_model(self.attr_func_cfg.n_epoch, is_snapshot=False)

    def _compute_attr_stats(self):
        """
        Compute the statistics of each attribute
        """
        try:
            attr_func = self.attr_func.module
        except AttributeError:
            attr_func = self.attr_func
        attr_info = compute_attr_stats(self.training_data, attr_func,
                                       verbose=False,
                                       rep=self.attr_rep_method)
        attr_info.update(self.training_data.attributes)
        # save attr information into the cfg
        OmegaConf.update(self.cfg, 'attr_func.encoded_attributes',
                         attr_info.to_dict(), force_add=True)

    def eval_epoch(self):
        loss_func = nn.CrossEntropyLoss(reduction='mean')
        eval_stats = eval_model(self.eval_data,
                                self.attr_func, loss_func,
                                self.cfg, self.eval_pred_thresholds, verbose=False)
        return eval_stats

    def train_batch(self, torch_attr, torch_traj_0, torch_mask_0,
                    torch_traj_1, torch_mask_1, torch_label, torch_sample_weights, mode='train'):
        if mode == 'train':
            self.attr_func.train()
        elif mode == 'eval':
            self.attr_func.eval()

        pred = self.attr_func(torch_attr, torch_traj_0, torch_traj_1,
                              torch_mask_0, torch_mask_1)
        _, traj_reward_0, _, traj_reward_1 = pred
        pred_rewards = torch.cat((traj_reward_0, traj_reward_1), dim=-1)
        loss = self.pred_loss(pred_rewards, torch_label)
        loss = torch.mean(loss * torch_sample_weights)

        if mode == "train":
            self.attr_func.zero_grad()
            loss.backward()
            self.optimizer.step()

        # loss info
        loss_info = {
            'loss': loss.data.cpu().numpy(),
            'avg_traj_reward': torch.mean(pred_rewards).data.cpu().numpy(),
        }

        # compute the prediction accuracy
        # ground truth
        np_target_labels = np.expand_dims(torch_label.data.cpu().numpy()[:, 0], axis=-1)
        ground_truth = np.zeros_like(np_target_labels).astype(np.int8)
        ground_truth[np_target_labels > 0.55] = 2
        ground_truth[np_target_labels < 0.45] = 1
        # prediction
        np_trace_reward_0, np_trace_reward_1 = traj_reward_0.data.cpu().numpy(), traj_reward_1.data.cpu().numpy()
        prob_pred = 1.0 / (1 + np.exp(np_trace_reward_1 - np_trace_reward_0))
        binary_pred = np.zeros_like(prob_pred).astype(np.int8)
        binary_pred[prob_pred > 0.55] = 2
        binary_pred[prob_pred < 0.45] = 1
        # accuracy
        acc = accuracy_score(ground_truth, binary_pred)
        loss_info.update({'accuracy': acc})

        return loss_info
