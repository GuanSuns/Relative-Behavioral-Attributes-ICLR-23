import itertools
import os
import pickle
import shutil
import time

import gym
from addict import Dict
from tqdm import tqdm
import numpy as np

from environments.highway.lane_change_env import Lane_Change_Env

DATA_SAVE_DIR = 'lane_change_data'


def main():
    from data.gen.highway_env.lane_change_synthetic_training import run_episode

    n_repeats = [i for i in range(1)]
    target_speeds = [(7.5, 8.5), (8.5, 9.5)]
    print(f'[INFO] n target speed: {len(target_speeds)}')
    turning_dists = [i for i in range(5, 31, 1)]
    print(f'[INFO] n turning dists: {len(turning_dists)}')
    tau_lateral = [0.2 + 0.2 * i for i in range(17)]
    print(f'[INFO] n tau lateral: {len(tau_lateral)}')

    experiments = list(itertools.product(n_repeats, target_speeds, turning_dists, tau_lateral))
    print(f'[INFO] {len(experiments)} experiments to run')

    experiment_records = []
    for e in tqdm(experiments):
        success, expr_name, expr_record = run_episode(target_speed=(e[1][0], e[1][1]), turning_dist=e[2],
                                                      tau_lateral=e[3], print_info=False, render=False)
        if success:
            # print(expr_record['traj_meta']['final_turning_dist'], len(expr_record['obs_sequence']))
            experiment_records.append(expr_record)

    video_path = f'{DATA_SAVE_DIR}/'
    os.makedirs(video_path, exist_ok=True)
    with open(f'{DATA_SAVE_DIR}/test_data.pickle', 'wb') as handle:
        pickle.dump(experiment_records, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

