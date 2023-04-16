import math
import random
import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from addict import Dict
from torch.utils.data import DataLoader

import architectures.utils
from utils import set_random_seed
from utils.logging_tools import Loss_Info


class Behavior_VAE_Trainer:
    def __init__(self, cfg, training_data, eval_data=None,
                 load_from=None, logger=None, log_freq=10, snapshot_freq=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        self.encoder_cfg = cfg.encoder_cfg
        self.logger = logger
        self.snapshot_freq = snapshot_freq

        if cfg.seed is None:
            cfg.seed = random.randint(1, 10000)
        set_random_seed(cfg.seed)

        if load_from is not None:
            from encoders.vae import get_trained_model
            self.behavior_vae, self.optimizer, self.cfg = get_trained_model(load_from, cfg, return_optimizer=True)
            self.encoder_cfg = self.cfg.encoder_cfg
        else:
            from encoders.vae.model import Behavior_VAE
            self.behavior_vae = Behavior_VAE(cfg).to(self.device)
            # init model
            self.behavior_vae.apply(architectures.utils.init_weights)
            self.optimizer = optim.Adam(self.behavior_vae.parameters(),
                                        weight_decay=self.encoder_cfg.weight_decay,
                                        lr=self.encoder_cfg.lr, betas=(0.9, 0.999))
        # parallel computing
        if torch.cuda.is_available():
            self.behavior_vae = torch.nn.DataParallel(self.behavior_vae)
        cudnn.benchmark = True  # optimize for fixed input size
        print('[INFO] total params: %.2fM' % (sum(p.numel() for p in self.behavior_vae.parameters()) / 1000000.0))

        # init scheduler
        if self.encoder_cfg.sche == "cosine":
            cosine_cfg = self.encoder_cfg.cosine_sche
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.encoder_cfg.n_epoch,
                                                                        eta_min=cosine_cfg.min_lr)
        elif self.encoder_cfg.sche == "step":
            step_cfg = self.encoder_cfg.step_sche
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=step_cfg.step_size,
                                                             gamma=step_cfg.gamma)
        elif self.encoder_cfg.sche == "const":
            self.scheduler = None
        else:
            raise NotImplementedError

        # load training data
        self.train_loader = DataLoader(training_data,
                                       num_workers=4,
                                       batch_size=self.encoder_cfg.batch_size,
                                       shuffle=True,
                                       drop_last=False,
                                       pin_memory=False)
        # load eval data
        self.eval_loader = None
        if eval_data is not None:
            self.eval_loader = DataLoader(eval_data,
                                          num_workers=4,
                                          batch_size=self.encoder_cfg.batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          pin_memory=False)
        self.log_freq = max(1, len(self.train_loader) // log_freq)

        # define loss
        self.epoch_loss_info = Loss_Info()
        self.batch_loss_info = Loss_Info()

    def save_model(self, epoch, is_snapshot=False):
        if self.logger is not None:
            try:
                model_state_dict = self.behavior_vae.module.state_dict()
            except AttributeError:
                model_state_dict = self.behavior_vae.state_dict()
            params = {
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
            }
            self.logger.save_models(checkpoint=params, cfg=self.cfg,
                                    is_snapshot=is_snapshot, postfix=str(epoch))

    def train(self):
        print(f'[INFO] start training for {self.encoder_cfg.n_epoch} epoch, # of mini-batch: {len(self.train_loader)}')
        n_batch_update = 0
        for i_epoch in range(self.encoder_cfg.n_epoch):
            if i_epoch and self.scheduler is not None:
                self.scheduler.step()

            self.epoch_loss_info.reset()
            for i, data in enumerate(self.train_loader):
                # torch_states : torch.Size([batch_size, frame_stack, 64, 64])
                torch_states = data.to(self.device)
                # model update
                losses_info = self.train_batch(torch_states, mode='train')
                n_batch_update += 1
                # save losses info
                self.epoch_loss_info.update(losses_info)
                self.batch_loss_info.update(losses_info)
                # logging: print info
                if i % 10 == 0 and i:
                    lr = self.optimizer.param_groups[0]['lr']
                    recon, kld_z = losses_info['loss_recon'], losses_info['kld_z']
                    print('[INFO] epoch %d, batch %d | recon: %.3f | kld_z: %.3f | lr: %.5f' % (
                               i_epoch, i, recon, kld_z, lr))
                # logging: to wandb or other logging system
                if i % self.log_freq == 0 and i:
                    if self.logger is not None:
                        lr = self.optimizer.param_groups[0]['lr']
                        log_info_dict = self.batch_loss_info.get_log_dict()
                        log_info_dict.update({'update_step': n_batch_update, 'epoch': i_epoch, 'lr': lr})
                        self.logger.log(log_info_dict)
                    self.batch_loss_info.reset()
            # get eval loss
            eval_loss_info = {}
            if self.eval_loader is not None:
                eval_loss_info = self.eval_epoch()

            # epoch logging
            log_info_dict = self.epoch_loss_info.get_log_dict()
            log_info_dict.update(eval_loss_info)
            print(f'[INFO] epoch {i_epoch} - {log_info_dict}')
            if self.logger is not None:
                lr = self.optimizer.param_groups[0]['lr']
                log_info_dict.update({'update_step': n_batch_update, 'epoch': i_epoch, 'lr': lr})
                self.logger.log(log_info_dict)
            self.epoch_loss_info.reset()

            # snapshot
            if i_epoch and self.snapshot_freq is not None and i_epoch % self.snapshot_freq == 0:
                self.save_model(i_epoch, is_snapshot=True)

        # save the trained model
        self.save_model(self.encoder_cfg.n_epoch, is_snapshot=False)

    def eval_epoch(self):
        eval_loss_info = Loss_Info()
        for i, data in enumerate(self.eval_loader):
            torch_states = data.to(self.device)
            with torch.no_grad():
                losses_info = self.train_batch(torch_states, mode='eval')
            eval_loss_info.update(losses_info)
        # rename loss info
        eval_loss_info = eval_loss_info.get_log_dict()
        renamed_eval_loss = Dict()
        for item in eval_loss_info:
            renamed_eval_loss[f'eval_{item}'] = eval_loss_info[item]
        return renamed_eval_loss

    def train_batch(self, torch_states, mode='train'):
        def get_recons_loss(_recons, _origin):
            if self.encoder_cfg.loss_recon == 'L2':
                return F.mse_loss(_recons, _origin, reduction='sum')
            else:
                return torch.abs(_recons - _origin).sum()

        if mode == 'train':
            self.behavior_vae.train()
        elif mode == 'eval':
            self.behavior_vae.eval()

        batch_size = torch_states.size(0)
        pred = self.behavior_vae(torch_states)  # pred
        z_mean, z_logvar, z_post, recon_x = pred

        # reconstruction loss
        l_recon = get_recons_loss(recon_x, torch_states)
        # KL losses
        kld_z = -0.5 * torch.sum(1 + z_logvar - torch.pow(z_mean, 2) - torch.exp(z_logvar))
        l_recon, kld_z = l_recon / batch_size, kld_z / batch_size

        loss = l_recon + kld_z * self.encoder_cfg.weight_z
        if mode == "train":
            self.behavior_vae.zero_grad()
            loss.backward()
            self.optimizer.step()

        # loss info
        loss_info = {
            'loss': loss.data.cpu().numpy(),
            'loss_recon': l_recon.data.cpu().numpy(),
            'kld_z': kld_z.data.cpu().numpy(),
        }
        return loss_info
