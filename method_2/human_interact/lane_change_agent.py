import itertools
import time

import cv2
import gym
import numpy as np
from addict import Dict
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from data.utils import get_attr_rep
from environments.highway.lane_change_env import Lane_Change_Env


class Lane_Change_Agent:
    """
    This Lane-Change agent uses model-based approach to
        optimize the provided reward.
    """

    # noinspection PyUnresolvedReferences
    def __init__(self, reward_func_path='trained_models/lane_change/method_2/'):
        self.env = gym.make("lanechange-v0", config={'manual_control': False})
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.reward_func_path = reward_func_path
        self.cfg = None
        self.reward_func = None
        self._load_models()

        # determine whether to use language attr representation
        self.attr_rep_method = 'one-hot'
        if 'use_language_attr' in self.cfg and self.cfg.use_language_attr:
            self.attr_rep_method = 'language'
        print(f'[INFO] attr representation method: {self.attr_rep_method}.')

        #################################################################
        ## Maintain a library of behaviors (for evaluation efficiency) ##
        #################################################################
        # policy config
        self.speeds = [(8.5, 9.5)]
        self.tau_lateral = [0.1 * i for i in range(1, 34)]
        self.turning_dist = [i for i in range(5, 31, 1)]

        from data.gen.highway_env.lane_change_rollout import Rollout_Generator
        rollout_generator = Rollout_Generator(encoder_path=self.cfg.reward_func.encoder_path)
        self.rollouts = rollout_generator.gen_rollouts(speeds=self.speeds,
                                                       turning_dists=self.turning_dist,
                                                       tau_laterals=self.tau_lateral)
        print(f'[INFO] generated {len(self.rollouts)} successful rollouts.')

        ###############################
        ## Belief of user preference ##
        ###############################
        print(f'[INFO] encoded attr info: {self.cfg.reward_func.encoded_attributes}')
        self.encoded_attr_info = self.cfg.reward_func.encoded_attributes
        # randomly pick an rollout as initial behavior
        self.curr_behavior_idx = np.random.randint(len(self.rollouts))
        self.curr_behavior = self.rollouts[self.curr_behavior_idx]

    def reset(self):
        # randomly pick an rollout as initial behavior
        self.curr_behavior_idx = np.random.randint(len(self.rollouts))
        self.curr_behavior = self.rollouts[self.curr_behavior_idx]

    def _load_models(self):
        # load default config
        default_cfg = OmegaConf.load('config/method_2/lane_change_env.yaml')
        # load the reward function first
        import method_2
        self.reward_func, _, self.cfg = method_2.get_trained_model(self.reward_func_path,
                                                                   default_cfg=default_cfg,
                                                                   return_optimizer=False)

    def get_query(self, **kwargs):
        """
        Making query according to current belief of user preference
        """
        return Dict(self.curr_behavior)

    def compute_rollout_reward(self, rollout_idx, attr_rep, anchor_traj, anchor_lens, to_increase):
        rollout_states = self.rollouts[rollout_idx].state_sequence
        self.reward_func.eval()
        with torch.no_grad():
            torch_traj = torch.from_numpy(rollout_states).float().to(self.device)
            _, rollout_reward = self.reward_func.reward(attr_rep, to_increase, anchor_traj,
                                                        anchor_lens, torch_traj)
        return rollout_reward.squeeze().detach().cpu().item()

    def _optimize_feedback(self, attr_name, to_increase):
        attr_rep = torch.from_numpy(
            get_attr_rep(attr_name, self.encoded_attr_info, self.attr_rep_method)).float().unsqueeze(dim=0).to(
            self.device)
        anchor_traj = torch.from_numpy(self.curr_behavior.state_sequence).float().unsqueeze(dim=0).to(self.device)
        anchor_lens = torch.tensor([len(self.curr_behavior.state_sequence)]).long().to(self.device)
        to_increase = torch.tensor([to_increase]).float().to(self.device)

        max_reward = -np.inf
        max_rollout_idx = 0
        for rollout_idx in range(len(self.rollouts)):
            rollout_reward = self.compute_rollout_reward(rollout_idx, attr_rep, anchor_traj,
                                                         anchor_lens, to_increase)
            if rollout_reward > max_reward:
                max_reward = rollout_reward
                max_rollout_idx = rollout_idx
        self.curr_behavior = Dict(self.rollouts[max_rollout_idx])

    def update_feedback(self, attr_feedback):
        """
        Update current behavior according to the user's feedback
        """
        prev_behavior_idx = int(self.curr_behavior_idx)
        is_satisfied = True
        for attr in attr_feedback:
            feedback = attr_feedback[attr]
            if (feedback > 0 and not self.encoded_attr_info[attr].reverse) or (
                    feedback < 0 and self.encoded_attr_info[attr].reverse):
                is_satisfied = False
                self._optimize_feedback(attr, 1.0)
            elif (feedback < 0 and not self.encoded_attr_info[attr].reverse) or (
                    feedback > 0 and self.encoded_attr_info[attr].reverse):
                is_satisfied = False
                self._optimize_feedback(attr, 0)


def main():
    agent = Lane_Change_Agent()
    query = agent.get_query()
    print(query.ground_truth_attr)


if __name__ == '__main__':
    main()
