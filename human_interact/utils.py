import os
import pprint
from pathlib import Path

import numpy as np
from addict import Dict
from omegaconf import OmegaConf
from data import get_attribute_data


def sample_eval_data(n_targets, dataset_name):
    """
    This function randomly pick feasible configurations from the test set
    """
    root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute())
    if dataset_name == 'lane-change':
        config_file = os.path.join(root_path, 'config/method_1/lane_change_env.yaml')
    elif dataset_name == 'manipulator':
        config_file = os.path.join(root_path, 'config/method_1/manipulator_lifting_env.yaml')
    elif dataset_name == 'walker':
        config_file = os.path.join(root_path, 'config/method_1/walker_step_env.yaml')
    else:
        raise NotImplementedError(f'{dataset_name}')

    # read config and data
    cfg = OmegaConf.load(config_file)

    _, attr_test_data = get_attribute_data(cfg, read_test_data=True)
    n_trajs = len(attr_test_data.behavior_data.traj_meta_info)
    print(f'[INFO] total # of trajectories: {n_trajs}.')
    assert n_trajs >= n_targets

    selected_samples = np.random.choice(n_trajs, n_targets)
    sample_infos = list()
    for i in range(n_targets):
        traj_info = attr_test_data.behavior_data.traj_meta_info[selected_samples[i]]
        sample_infos.append(Dict(traj_info))
    return sample_infos


def main():
    n_targets = 20
    dataset_name = 'lane-change'

    attributes = dict()
    if dataset_name == 'lane-change':
        attributes = {'sharpness': 'tau_lateral', 'turning_dist': 'final_turning_dist'}
    if dataset_name == 'walker':
        attributes = {'step_size': 'final_step_size', 'softness': 'softness_speed'}
    if dataset_name == 'manipulator':
        attributes = {'hardness': 'hardness', 'force': 'force'}

    sample_infos = sample_eval_data(n_targets, dataset_name)

    # extract attribute info
    target_attributes = Dict()
    for attr in attributes:
        target_attributes[attr] = list()
        for sample_idx in range(len(sample_infos)):
            target_attributes[attr].append(sample_infos[sample_idx][attributes[attr]])
    # print info
    for attr in attributes:
        print(f'{attr} | {target_attributes[attr]}')


if __name__ == '__main__':
    main()
