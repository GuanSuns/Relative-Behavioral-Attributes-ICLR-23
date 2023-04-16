import random

import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from addict import Dict
from torch.utils.data import DataLoader

from method_1.reward_func.data import Traj_Pref_Data
import architectures.utils
from utils import set_random_seed
from utils.logging_tools import Loss_Info


class Reward_Func_Trainer:
    def __init__(self, cfg, training_attr_data, eval_attr_data=None, load_from=None,
                 logger=None, log_freq=10, snapshot_freq=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        self.reward_func_cfg = self.cfg.reward_func
        self.logger = logger
        self.snapshot_freq = snapshot_freq

        if cfg.seed is None:
            cfg.seed = random.randint(1, 10000)
        set_random_seed(cfg.seed)

        if load_from is not None:
            from method_1.reward_func import get_trained_model
            self.reward_func, self.optimizer, self.cfg = get_trained_model(load_from, cfg, return_optimizer=True)
            self.reward_func_cfg = self.cfg.reward_func
        else:
            from method_1.reward_func.model import Reward_Func
            self.reward_func = Reward_Func(cfg).to(self.device)
            # init model
            self.reward_func.apply(architectures.utils.init_weights)
            self.optimizer = optim.Adam(self.reward_func.parameters(),
                                        lr=self.reward_func_cfg.lr, betas=(0.9, 0.999),
                                        weight_decay=self.reward_func_cfg.weight_decay)
        # parallel computing
        if torch.cuda.is_available():
            self.reward_func = torch.nn.DataParallel(self.reward_func)
        # cudnn.benchmark = True  # optimize for fixed input size
        print('[INFO] total params: %.2fM' % (sum(p.numel() for p in self.reward_func.parameters()) / 1000000.0))

        # init scheduler
        if self.reward_func_cfg.sche == "cosine":
            cosine_cfg = self.reward_func_cfg.cosine_sche
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.reward_func_cfg.n_epoch,
                                                                        eta_min=cosine_cfg.min_lr)
        elif self.reward_func_cfg.sche == "step":
            step_cfg = self.reward_func_cfg.step_sche
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=step_cfg.step_size,
                                                             gamma=step_cfg.gamma)
        elif self.reward_func_cfg.sche == "const":
            self.scheduler = None
        else:
            raise NotImplementedError

        # turn attribute data into traj-preference dataset
        self.traj_pref_training = Traj_Pref_Data(training_attr_data, self.cfg.reward_func.n_targets_per_pair, self.cfg)
        self.traj_pref_test = Traj_Pref_Data(eval_attr_data, self.cfg.reward_func.n_targets_per_pair_eval, self.cfg) if eval_attr_data else None

        # load training data
        self.train_loader = DataLoader(self.traj_pref_training,
                                       num_workers=4,
                                       batch_size=self.reward_func_cfg.batch_size,
                                       shuffle=True,
                                       drop_last=False,
                                       pin_memory=False)
        # load eval data
        self.eval_loader = None
        if eval_attr_data is not None:
            self.eval_loader = DataLoader(self.traj_pref_test,
                                          num_workers=4,
                                          batch_size=self.reward_func_cfg.eval_batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          pin_memory=False)
        self.log_freq = max(1, len(self.train_loader) // log_freq)
        self.print_log_freq = max(1, len(self.train_loader) // 10)

        # define loss
        self.pred_loss = nn.CrossEntropyLoss(reduction='mean')
        self.epoch_loss_info = Loss_Info()
        self.batch_loss_info = Loss_Info()

    def save_model(self, epoch, is_snapshot=False):
        if self.logger is not None:
            try:
                model_state_dict = self.reward_func.module.state_dict()
            except AttributeError:
                model_state_dict = self.reward_func.state_dict()
            params = {
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
            }
            self.logger.save_models(checkpoint=params, cfg=self.cfg,
                                    is_snapshot=is_snapshot, postfix=str(epoch))

    def train(self):
        print(f'[INFO] start training for {self.reward_func_cfg.n_epoch} epoch, # of mini-batch: {len(self.train_loader)}')
        n_batch_update = 0
        for epoch in range(self.reward_func_cfg.n_epoch):
            if epoch and self.scheduler is not None:
                self.scheduler.step()

            self.epoch_loss_info.reset()

            for i, data in enumerate(self.train_loader):
                # torch_traj_0 : torch.Size([batch_size, trace_len, z_dim])
                # torch_mask_0: torch.Size([batch_size, trace_len])
                # torch_traj_1: torch.Size([batch_size, trace_len, z_dim])
                # torch_mask_1: torch.Size([batch_size, trace_len])
                # torch_target_attrs: torch.Size([batch_size, attr_dim])
                # torch_label: torch.Size([batch_size, 2])
                torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, torch_target_attrs, torch_label = data
                torch_target_attrs = torch_target_attrs.to(self.device)
                torch_traj_0 = torch_traj_0.to(self.device)
                torch_mask_0 = torch_mask_0.to(self.device)
                torch_traj_1 = torch_traj_1.to(self.device)
                torch_mask_1 = torch_mask_1.to(self.device)
                torch_label = torch_label.to(self.device)
                # model update
                losses_info = self.train_batch(torch_target_attrs, torch_traj_0, torch_mask_0,
                                               torch_traj_1, torch_mask_1,
                                               torch_label, mode='train')
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
                if i % self.log_freq == 0 and i:
                    if self.logger is not None:
                        lr = self.optimizer.param_groups[0]['lr']
                        log_info_dict = self.batch_loss_info.get_log_dict()
                        log_info_dict.update({'update_step': n_batch_update, 'epoch': epoch, 'lr': lr})
                        self.logger.log(log_info_dict)
                    self.batch_loss_info.reset()
            # get eval loss
            eval_loss_info = {}
            if self.eval_loader is not None:
                eval_loss_info = self.eval_epoch()

            # epoch logging
            log_info_dict = self.epoch_loss_info.get_log_dict()
            log_info_dict.update(eval_loss_info)
            print(f'[INFO] epoch {epoch} - {log_info_dict}')
            if self.logger is not None:
                lr = self.optimizer.param_groups[0]['lr']
                log_info_dict.update({'update_step': n_batch_update, 'epoch': epoch, 'lr': lr})
                self.logger.log(log_info_dict)
            self.epoch_loss_info.reset()

            # snapshot
            if epoch and self.snapshot_freq is not None and epoch % self.snapshot_freq == 0:
                self.save_model(epoch, is_snapshot=True)

        # save the trained model
        self.save_model(self.reward_func_cfg.n_epoch, is_snapshot=False)

    def eval_epoch(self):
        eval_loss_info = Loss_Info()
        for i, data in enumerate(self.eval_loader):
            torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, torch_target_attrs, torch_label = data
            torch_target_attrs = torch_target_attrs.to(self.device)
            torch_traj_0 = torch_traj_0.to(self.device)
            torch_mask_0 = torch_mask_0.to(self.device)
            torch_traj_1 = torch_traj_1.to(self.device)
            torch_mask_1 = torch_mask_1.to(self.device)
            torch_label = torch_label.to(self.device)
            losses_info = self.train_batch(torch_target_attrs, torch_traj_0, torch_mask_0,
                                           torch_traj_1, torch_mask_1,
                                           torch_label, mode='eval')
            eval_loss_info.update(losses_info)
        # rename loss info
        eval_loss_info = eval_loss_info.get_log_dict()
        renamed_eval_loss = Dict()
        for item in eval_loss_info:
            renamed_eval_loss[f'eval_{item}'] = eval_loss_info[item]
        return renamed_eval_loss

    def train_batch(self, torch_target_attrs, torch_traj_0, torch_mask_0,
                    torch_traj_1, torch_mask_1, torch_label, mode='train'):
        if mode == 'train':
            self.reward_func.train()
        elif mode == 'eval':
            self.reward_func.eval()

        pred = self.reward_func(torch_target_attrs, torch_traj_0, torch_traj_1,
                                torch_mask_0, torch_mask_1)
        _, traj_reward_0, _, traj_reward_1 = pred
        pred_rewards = torch.cat((traj_reward_0, traj_reward_1), dim=-1)
        loss = self.pred_loss(pred_rewards, torch_label)

        if mode == "train":
            self.reward_func.zero_grad()
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
