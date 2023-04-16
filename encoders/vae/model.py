import torch.nn as nn
import torch

from architectures.cnn import Encoder, LinearUnit, Decoder


class Behavior_VAE(nn.Module):
    """
    This behavior encoder operates on stacked grayscale images
    """
    def __init__(self, cfg):
        super(Behavior_VAE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.z_dim = cfg.encoder_cfg.z_dim  # default: 64
        self.cnn_output_dim = cfg.encoder_cfg.cnn_output_dim  # default: 512
        self.hidden_dim = cfg.encoder_cfg.hidden_dim  # default: 256
        self.frame_stack = cfg.frame_stack  # number of frames in a video

        # Encoder
        self.encoder = Encoder(self.cnn_output_dim, self.frame_stack)
        self.z_mlp = LinearUnit(self.cnn_output_dim, self.hidden_dim, False)
        self.z_mean = LinearUnit(self.hidden_dim, self.z_dim, False)
        self.z_logvar = LinearUnit(self.hidden_dim, self.z_dim, False)

        # Decoder
        self.decoder = Decoder(self.z_dim, self.frame_stack)

    def encode_and_sample_post(self, x):
        """
        :param x: torch.Size([batch_size, 3, size, size])
        """
        conv_x = self.encoder(x)[0]
        out_z = self.z_mlp(conv_x)
        z_mean = self.z_mean(out_z)
        z_logvar = self.z_logvar(out_z)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=True)
        return z_mean, z_logvar, z_post

    def forward(self, x):
        z_mean, z_logvar, z_post = self.encode_and_sample_post(x)
        recon_x = self.decoder(z_post)
        return z_mean, z_logvar, z_post, recon_x

    def forward_prior_and_post(self, x):
        z_mean, z_logvar, z_post = self.encode_and_sample_post(x)
        # decode with prior f
        z_prior = self.reparameterize(torch.zeros(z_post.shape).to(self.device),
                                      torch.zeros(z_post.shape).to(self.device), random_sampling=True)
        recon_x_prior = self.decoder(z_prior)
        # decode with post f
        recon_x_post = self.decoder(z_post)
        return recon_x_prior, recon_x_post

    def get_behavior_emb(self, x):
        """
        Input shape: [batch_size, frame_stack, size, size] or [frame_stack, size, size]
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        z_mean, _, _ = self.encode_and_sample_post(x)
        return z_mean

    @staticmethod
    def reparameterize(mean, logvar, random_sampling=True):
        """
        Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        """
        if random_sampling is True:
            eps = torch.randn_like(logvar)  # normal distribution
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

