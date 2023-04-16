import numpy as np
import torch
import torch.nn as nn

from method_1.reward_func.utils import to_attrs_vec


class Reward_Func(nn.Module):
    def __init__(self, cfg):
        super(Reward_Func, self).__init__()
        self.cfg = cfg
        self.reward_clip = cfg.reward_func.reward_clip
        self.state_dim = cfg.reward_func.state_dim      # the dimension of state representation
        self.attr_latent_dim = len([_attr_name for _attr_name in self.cfg.attr_info.attributes])
        self.hidden_dim = cfg.reward_func.hidden_dim

        self.reward_func = nn.Sequential(
            nn.Linear(self.state_dim + self.attr_latent_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # attribute information
        self.encoded_attr_info = self.cfg.attr_func.encoded_attributes

    def reward_target_attr(self, target_attr, traces, trace_masks=None):
        """
        traces: torch.Size([batch_size, trace_len, emb_size]) or torch.Size([trace_len, emb_size])
        trace_masks is used when traces are in different lengths:
                    torch.Size([batch_size, trace_len]) or torch.Size([trace_len])
        target_attr: a list of dictionary that contains raw/un-normalized target attribute scores
        """
        if not isinstance(target_attr, list):
            target_attr = [target_attr]
        target_attrs_vec = to_attrs_vec(target_attr, self.encoded_attr_info)
        target_attrs_vec = torch.from_numpy(target_attrs_vec).float().to(traces.device)
        return self.reward(target_attrs_vec, traces, trace_masks=trace_masks)

    def reward(self, target_attrs_vec, traces, trace_masks=None):
        """
        traces: torch.Size([batch_size, trace_len, emb_size]) or torch.Size([trace_len, emb_size])
        trace_masks is used when traces are in different lengths:
            torch.Size([batch_size, trace_len]) or torch.Size([trace_len])
        target_attrs_vec: torch.Size([batch_size, attr_dim]) or torch.Size([attr_dim])
        """
        # preprocessing traces and masks
        if len(traces.shape) == 2:
            traces = traces.unsqueeze(0)
        if trace_masks is not None and len(trace_masks.shape) == 1:
            trace_masks = trace_masks.unsqueeze(0)
        if len(target_attrs_vec.shape) == 1:
            target_attrs_vec = target_attrs_vec.unsqueeze(0)
        batch_size, trace_len = traces.size(0), traces.size(1)

        # reshape to torch.Size([batch_size * trace_len, emb_size])
        traces = traces.view(-1, traces.shape[-1])
        # repeat attrs to [batch_size * trace_len, emb_size] append attribute to traces
        traces_attrs_vector = torch.cat((traces, torch.repeat_interleave(target_attrs_vec, repeats=trace_len, dim=0)), dim=-1)
        # compute the reward for each state in the traces
        state_rewards = self.reward_func(traces_attrs_vector)
        state_rewards = state_rewards.view(batch_size, trace_len, -1)
        # mask if not of the same length
        if trace_masks is not None:
            state_rewards = state_rewards * trace_masks.unsqueeze(-1)
        # clip trace reward
        trace_reward = torch.clip(torch.sum(state_rewards, 1), min=-self.reward_clip, max=self.reward_clip)
        return state_rewards, trace_reward

    def predict_prob(self, target_attr_vec, trace_0, trace_1, trace_mask_0=None, trace_mask_1=None):
        """
        Compute P[t_0 > t_1] = exp[sum(r(t_0))]/{exp[sum(r(t_0))]+exp[sum(r(t_1))]}
                = 1 /{1+exp[sum(r(t_1) - r(t_0))]}
        """
        _, trace_reward_0 = self.reward(target_attr_vec, trace_0, trace_mask_0)
        _, trace_reward_1 = self.reward(target_attr_vec, trace_1, trace_mask_1)
        r1_minus_r0 = trace_reward_1 - trace_reward_0
        prob = 1.0 / (1.0 + torch.exp(r1_minus_r0))
        return prob

    def forward(self, target_attr_vec, trace_0, trace_1, trace_mask_0=None, trace_mask_1=None):
        state_rewards_0, trace_reward_0 = self.reward(target_attr_vec, trace_0, trace_mask_0)
        state_rewards_1, trace_reward_1 = self.reward(target_attr_vec, trace_1, trace_mask_1)
        return state_rewards_0, trace_reward_0, state_rewards_1, trace_reward_1


def main():
    pass


if __name__ == '__main__':
    main()


