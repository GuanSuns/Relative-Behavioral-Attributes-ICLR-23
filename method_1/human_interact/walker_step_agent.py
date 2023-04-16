import itertools
import os

import cv2
import gym
import numpy as np
from addict import Dict
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


class Walker_Step_Agent:
    """
    This Walker-Step agent uses multiple pretrained policies to optimize the user reward
    """

    # noinspection PyUnresolvedReferences
    def __init__(self, reward_func_path='trained_models/walker_step/method_1/reward_func'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.reward_func_path = reward_func_path
        self.cfg = None
        self.attr_func = None
        self.reward_func = None
        self._load_models()

        #################################################################
        ## Maintain a library of behaviors (for evaluation efficiency) ##
        #################################################################
        file_dir = 'data/gen/walker_step'
        self.policy_model_list = [
            os.path.join(file_dir, 'step_policy/td3_step_model.zip'),
            os.path.join(file_dir, 'step_policy/td3_step_model.zip'),
        ]

        # policy config
        self.step_sizes = [0.1 + i * 0.05 for i in range(0, 23)]  # 0.1 to 1.2, step: 0.1, high precision: 0.02
        self.speeds = [0.005 + 0.013 * i for i in range(0, 16)]  # 0.005 to 0.2, step: 0.013

        from data.gen.walker_step.walker_step_rollout import Rollout_Generator
        rollout_generator = Rollout_Generator(model_paths=self.policy_model_list)
        self.rollouts = rollout_generator.gen_rollouts(self.step_sizes, self.speeds)
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
        self.encoded_attr_info = self.cfg.attr_func.encoded_attributes
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
        default_cfg = OmegaConf.load('config/method_1/walker_step_env.yaml')
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

    def compute_rollout_reward(self, rollout_idx, target_attr):
        rollout_states = np.array(self.rollouts[rollout_idx]['state_sequence'])
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
    agent = Walker_Step_Agent()
    query = agent.get_query()
    print(query.ground_truth_attr)


if __name__ == '__main__':
    main()
