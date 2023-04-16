import torch
import torch.nn as nn


class Attr_Func(nn.Module):
    def __init__(self, cfg):
        super(Attr_Func, self).__init__()
        self.cfg = cfg
        self.reward_clip = cfg.attr_func.reward_clip
        self.state_dim = cfg.attr_func.state_dim  # the dimension of state representation
        self.attr_dim = cfg.attr_func.attr_dim  # the dimension of attribute representation
        self.hidden_dim = cfg.attr_func.hidden_dim

        self.ensemble_size = cfg.attr_func.n_ensemble
        self.reward_func_ensemble = nn.ModuleList([nn.Sequential(
            nn.Linear(self.state_dim + self.attr_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        ) for _ in range(self.ensemble_size)])

        self.reward_func = None if self.ensemble_size > 1 else self.reward_func_ensemble[0]

    def reward_ensemble(self, attrs, traces, trace_masks=None, ensemble_idx=0):
        """
        traces: torch.Size([batch_size, trace_len, emb_size]) or torch.Size([trace_len, emb_size])
        trace_masks is used when traces are in different lengths:
            torch.Size([batch_size, trace_len]) or torch.Size([trace_len])
        attr: torch.Size([batch_size, attr_dim]) or torch.Size([attr_dim])
        """
        reward_func = self.reward_func_ensemble[ensemble_idx]
        # preprocessing traces and masks
        if len(traces.shape) == 2:
            traces = traces.unsqueeze(0)
        if trace_masks is not None and len(trace_masks.shape) == 1:
            trace_masks = trace_masks.unsqueeze(0)
        if len(attrs.shape) == 1:
            attrs = attrs.unsqueeze(0)
        batch_size, trace_len = traces.size(0), traces.size(1)

        # reshape to torch.Size([batch_size * trace_len, emb_size])
        traces = traces.view(-1, traces.shape[-1])
        # repeat attrs to [batch_size * trace_len, emb_size] append attribute to traces
        traces_attrs = torch.cat((traces, torch.repeat_interleave(attrs, repeats=trace_len, dim=0)), dim=-1)
        # compute the reward for each state in the traces
        state_rewards = reward_func(traces_attrs)
        state_rewards = state_rewards.view(batch_size, trace_len, -1)
        # mask if not of the same length
        if trace_masks is not None:
            state_rewards = state_rewards * trace_masks.unsqueeze(-1)
        # clip trace reward
        trace_reward = torch.clip(torch.sum(state_rewards, 1), min=-self.reward_clip, max=self.reward_clip)
        return state_rewards, trace_reward

    def reward(self, attrs, traces, trace_masks=None):
        ensemble_state_rew, ensemble_trace_rew = 0, 0
        for ensemble_idx in range(self.ensemble_size):
            state_rewards, trace_reward = self.reward_ensemble(attrs, traces,
                                                               trace_masks=trace_masks,
                                                               ensemble_idx=ensemble_idx)
            ensemble_state_rew += state_rewards
            ensemble_trace_rew += trace_reward
        return ensemble_state_rew / float(self.ensemble_size), ensemble_trace_rew / float(self.ensemble_size)

    def reward_with_confidence(self, attrs, traces, trace_masks=None):
        state_rewards_list, trace_reward_list = list(), list()
        for ensemble_idx in range(self.ensemble_size):
            state_rewards, trace_reward = self.reward_ensemble(attrs, traces,
                                                               trace_masks=trace_masks,
                                                               ensemble_idx=ensemble_idx)
            state_rewards_list.append(state_rewards)
            trace_reward_list.append(trace_reward)

        ensemble_state_rew, ensemble_trace_rew = 0, 0
        for ensemble_idx in range(self.ensemble_size):
            ensemble_state_rew += state_rewards_list[ensemble_idx]
            ensemble_trace_rew += trace_reward_list[ensemble_idx]

        state_rewards = ensemble_state_rew / float(self.ensemble_size)
        trace_rewards = ensemble_trace_rew / float(self.ensemble_size)
        confidences = torch.var(torch.stack(trace_reward_list), dim=0)

        return state_rewards, trace_rewards, confidences

    def predict_prob(self, attrs, trace_0, trace_1, trace_mask_0=None, trace_mask_1=None):
        """
        Compute P[t_0 > t_1] = exp[sum(r(t_0))]/{exp[sum(r(t_0))]+exp[sum(r(t_1))]}
                = 1 /{1+exp[sum(r(t_1) - r(t_0))]}
        """
        _, trace_reward_0 = self.reward(attrs, trace_0, trace_mask_0)
        _, trace_reward_1 = self.reward(attrs, trace_1, trace_mask_1)
        r1_minus_r0 = trace_reward_1 - trace_reward_0
        prob = 1.0 / (1.0 + torch.exp(r1_minus_r0))
        return prob

    def forward(self, attrs, trace_0, trace_1, trace_mask_0=None, trace_mask_1=None):
        state_rewards_0, trace_reward_0 = self.reward(attrs, trace_0, trace_mask_0)
        state_rewards_1, trace_reward_1 = self.reward(attrs, trace_1, trace_mask_1)
        return state_rewards_0, trace_reward_0, state_rewards_1, trace_reward_1


def main():
    pass


if __name__ == '__main__':
    main()
