import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from method_1.reward_func.utils import to_attrs_vec


class Reward_Func(nn.Module):
    def __init__(self, cfg):
        super(Reward_Func, self).__init__()
        self.cfg = cfg
        self.reward_clip = cfg.reward_func.reward_clip

        self.state_dim = cfg.reward_func.state_dim  # the dimension of state representation
        self.anchor_hidden_dim = cfg.reward_func.anchor_hidden_dim
        self.anchor_emb_dim = cfg.reward_func.anchor_emb_dim
        self.anchor_encoder = nn.LSTM(input_size=self.state_dim, hidden_size=self.anchor_hidden_dim, num_layers=2,
                                      bidirectional=True, batch_first=True)
        self.anchor_encoder_mlp = nn.Sequential(
            nn.Linear(self.anchor_hidden_dim, self.anchor_emb_dim),
            nn.LeakyReLU()
        )

        self.attr_latent_dim = cfg.reward_func.attr_dim  # the dimension of attribute latent code
        self.hidden_dim = cfg.reward_func.hidden_dim
        self.reward_func = nn.Sequential(
            nn.Linear(self.state_dim + self.attr_latent_dim + 1 + self.anchor_emb_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def encode_anchor(self, anchor_trace, anchor_lens):
        self.anchor_encoder.flatten_parameters()

        if len(anchor_trace.shape) == 2:
            anchor_trace = anchor_trace.unsqueeze(0)
        anchor_lens = anchor_lens.cpu().numpy()
        batch_size = anchor_trace.size(0)
        assert batch_size == anchor_lens.shape[0], f'batch_size:{batch_size}|anchor_lens:{anchor_lens.shape[0]}'

        # pack the padded anchor_trace
        packed_input = pack_padded_sequence(anchor_trace, anchor_lens, batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.anchor_encoder(packed_input)
        padded_output, _ = pad_packed_sequence(packed_lstm_out, batch_first=True, total_length=np.max(anchor_lens))
        # extract anchor embeddings
        lstm_b_outputs = list()
        for anchor_i in range(batch_size):
            n_frames = anchor_lens[anchor_i]
            lstm_out = padded_output[anchor_i, :n_frames, :].unsqueeze(0)
            # get trace embedding
            backward = lstm_out[:, 0, self.anchor_hidden_dim: 2 * self.anchor_hidden_dim]
            frontal = lstm_out[:, n_frames - 1, 0:self.anchor_hidden_dim]
            lstm_out_b = (backward + frontal) / 2
            lstm_b_outputs.append(lstm_out_b)
        lstm_out_b = torch.cat(lstm_b_outputs, dim=0)
        # get behavior emb
        anchor_embs = self.anchor_encoder_mlp(lstm_out_b)
        return anchor_embs

    def reward(self, attr, to_increase, anchor_trace,
               anchor_trace_len, traces, trace_masks=None):
        """
        attr: torch.Size([batch_size, attr_latent_dim]) or torch.Size([attr_latent_dim, ])
        anchor_trace: torch.Size([batch_size, trace_len, emb_size]) or torch.Size([trace_len, emb_size])
        traces: torch.Size([batch_size, trace_len, emb_size]) or torch.Size([trace_len, emb_size])
        trace_masks is used when traces are in different lengths:
            torch.Size([batch_size, trace_len]) or torch.Size([trace_len])
        """
        # preprocessing traces and masks
        if len(attr.shape) == 1:
            attr = attr.unsqueeze(0)
        if len(to_increase.shape) == 1:
            to_increase = to_increase.unsqueeze(0)
        if len(anchor_trace.shape) == 2:
            anchor_trace = anchor_trace.unsqueeze(0)
        if len(traces.shape) == 2:
            traces = traces.unsqueeze(0)
        if trace_masks is not None and len(trace_masks.shape) == 1:
            trace_masks = trace_masks.unsqueeze(0)

        batch_size, trace_len = traces.size(0), traces.size(1)

        # encode anchor trace
        anchor_emb = self.encode_anchor(anchor_trace, anchor_trace_len)
        # reshape to torch.Size([batch_size * trace_len, emb_size])
        traces = traces.view(-1, traces.shape[-1])
        # append to_increase to attr
        anchor_attr_direction = torch.cat((attr, to_increase, anchor_emb), dim=-1)
        # repeat attrs to [batch_size * trace_len, size of anchor_attr_direction] append attribute to traces
        traces_attrs = torch.cat((traces, torch.repeat_interleave(anchor_attr_direction, repeats=trace_len, dim=0)), dim=-1)
        # compute the reward for each state in the traces
        state_rewards = self.reward_func(traces_attrs)
        state_rewards = state_rewards.view(batch_size, trace_len, -1)
        # mask if not of the same length
        if trace_masks is not None:
            state_rewards = state_rewards * trace_masks.unsqueeze(-1)
        # clip trace reward
        trace_reward = torch.clip(torch.sum(state_rewards, 1), min=-self.reward_clip, max=self.reward_clip)
        return state_rewards, trace_reward

    def predict_prob(self, attr, to_increase, anchor_trace, anchor_trace_len,
                     trace_0, trace_1, trace_mask_0=None, trace_mask_1=None):
        """
        Compute P[t_0 > t_1] = exp[sum(r(t_0))]/{exp[sum(r(t_0))]+exp[sum(r(t_1))]}
                = 1 /{1+exp[sum(r(t_1) - r(t_0))]}
        """
        _, trace_reward_0 = self.reward(attr, to_increase, anchor_trace, anchor_trace_len,
                                        trace_0, trace_mask_0)
        _, trace_reward_1 = self.reward(attr, to_increase, anchor_trace, anchor_trace_len,
                                        trace_1, trace_mask_1)
        r1_minus_r0 = trace_reward_1 - trace_reward_0
        prob = 1.0 / (1.0 + torch.exp(r1_minus_r0))
        return prob

    def forward(self, attr, to_increase, anchor_trace, anchor_trace_len,
                trace_0, trace_1, trace_mask_0=None, trace_mask_1=None):
        state_rewards_0, trace_reward_0 = self.reward(attr, to_increase, anchor_trace, anchor_trace_len,
                                                      trace_0, trace_mask_0)
        state_rewards_1, trace_reward_1 = self.reward(attr, to_increase, anchor_trace, anchor_trace_len,
                                                      trace_1, trace_mask_1)
        return state_rewards_0, trace_reward_0, state_rewards_1, trace_reward_1


def main():
    pass


if __name__ == '__main__':
    main()


