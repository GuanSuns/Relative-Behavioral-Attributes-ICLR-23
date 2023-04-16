import itertools
import time

import cv2
import gym
import numpy as np
from addict import Dict
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from data.utils import resize_sequence, get_attr_rep
from environments.highway.lane_change_env import Lane_Change_Env


class Lane_Change_Agent:
    """
    This Lane-Change agent uses model-based approach to
        optimize the provided reward.
    """
    # noinspection PyUnresolvedReferences
    def __init__(self, reward_func_path='trained_models/lane_change/method_1/reward_func'):
        self.env = gym.make("lanechange-v0", config={'manual_control': False})
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.reward_func_path = reward_func_path
        self.cfg = None
        self.encoder = None
        self.attr_func = None
        self.reward_func = None
        self._load_models()

        #################################################################
        ## Maintain a library of behaviors (for evaluation efficiency) ##
        #################################################################
        # policy config
        self.speeds = [(8.5, 9.5)]
        self.tau_lateral = [0.1 * i for i in range(1, 35)]
        self.turning_dist = [i for i in range(5, 31, 1)]

        from data.gen.highway_env.lane_change_rollout import Rollout_Generator
        rollout_generator = Rollout_Generator(encoder_path=self.cfg.attr_func.encoder_path)
        self.rollouts = rollout_generator.gen_rollouts(speeds=self.speeds,
                                                       turning_dists=self.turning_dist,
                                                       tau_laterals=self.tau_lateral)
        print(f'[INFO] generated {len(self.rollouts)} successful rollouts.')

        ###############################
        ## Belief of user preference ##
        ###############################
        print(f'[INFO] encoded attr info: {self.cfg.attr_func.encoded_attributes}')
        self.encoded_attr_info = self.cfg.attr_func.encoded_attributes
        self.curr_attr_belief = Dict()
        self.attr_belief_upper = Dict()
        self.attr_belief_lower = Dict()
        for attr in self.encoded_attr_info:
            attr_min, attr_max = self.encoded_attr_info[attr].min, self.encoded_attr_info[attr].max
            self.attr_belief_lower[attr] = attr_min
            self.attr_belief_upper[attr] = attr_max
            self.curr_attr_belief[attr] = (attr_min + attr_max) / 2.0

    def reset(self):
        # reset belief of user preference
        self.curr_attr_belief = Dict()
        self.attr_belief_upper = Dict()
        self.attr_belief_lower = Dict()
        for attr in self.encoded_attr_info:
            attr_min, attr_max = self.encoded_attr_info[attr].min, self.encoded_attr_info[attr].max
            self.attr_belief_lower[attr] = attr_min
            self.attr_belief_upper[attr] = attr_max
            self.curr_attr_belief[attr] = (attr_min + attr_max) / 2.0

    def _load_models(self):
        # load default config
        default_cfg = OmegaConf.load('config/method_1/lane_change_env.yaml')
        # load the reward function first
        import method_1.reward_func
        self.reward_func, _, self.cfg = method_1.reward_func.get_trained_model(self.reward_func_path,
                                                                               default_cfg=default_cfg,
                                                                               return_optimizer=False)
        # load the attribute function
        import method_1.attr_func
        self.attr_func, _, self.cfg = method_1.attr_func.get_trained_model(self.cfg.reward_func.attr_func_path,
                                                                           default_cfg=self.cfg,
                                                                           return_optimizer=False)
        # load the encoder
        import encoders
        self.encoder, _, self.cfg = encoders.get_trained_encoder(self.cfg.attr_func.encoder_path,
                                                                 default_cfg=self.cfg,
                                                                 return_optimizer=False)

    def compute_rollout_reward(self, rollout_idx, target_attr):
        rollout_states = self.rollouts[rollout_idx].state_sequence
        self.reward_func.eval()
        with torch.no_grad():
            torch_traj = torch.from_numpy(rollout_states).float().to(self.device)
            _, rollout_reward = self.reward_func.reward_target_attr(target_attr, torch_traj)
        return rollout_reward.squeeze().detach().cpu().item()

    def optimize_reward(self, target_attr, **kwargs):
        max_reward = -np.inf
        max_rollout_idx = 0
        for rollout_idx in range(len(self.rollouts)):
            rollout_reward = self.compute_rollout_reward(rollout_idx, target_attr)
            if rollout_reward > max_reward:
                max_reward = rollout_reward
                max_rollout_idx = rollout_idx
        # return rollout config
        return Dict(self.rollouts[max_rollout_idx])

    def get_query(self, **kwargs):
        """
        Making query according to current belief of user preference
        """
        # get target attr vector according to current belief
        return self.optimize_reward(self.curr_attr_belief, **kwargs)

    def update_feedback(self, attr_feedback):
        """
        Use binary search to find user preferred attribute score
        """
        for attr in attr_feedback:
            feedback = attr_feedback[attr]
            if (feedback > 0 and not self.encoded_attr_info[attr].reverse) or (
                    feedback < 0 and self.encoded_attr_info[attr].reverse):
                # the user wants to increase the strength of attr
                self.attr_belief_lower[attr] = self.curr_attr_belief[attr]
                self.curr_attr_belief[attr] = (self.attr_belief_lower[attr] + self.attr_belief_upper[attr]) / 2.0
            elif (feedback < 0 and not self.encoded_attr_info[attr].reverse) or (
                    feedback > 0 and self.encoded_attr_info[attr].reverse):
                # the user wants to decrease the strength of attr
                self.attr_belief_upper[attr] = self.curr_attr_belief[attr]
                self.curr_attr_belief[attr] = (self.attr_belief_lower[attr] + self.attr_belief_upper[attr]) / 2.0
        print(f'[INFO] updated attr belief: {self.curr_attr_belief}')
        print(f'[INFO] upper: {self.attr_belief_upper}.')
        print(f'[INFO] lower: {self.attr_belief_lower}.')
        print('-' * 20)


def main():
    agent = Lane_Change_Agent()
    query = agent.get_query()
    print(query.ground_truth_attr)


if __name__ == '__main__':
    main()
