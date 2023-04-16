import itertools
import os
import pickle

import numpy as np
from addict import Dict
from stable_baselines3 import TD3
from tqdm import tqdm

from environments.walker.walker_step import Walker_Step_Env

DATA_SAVE_DIR = 'walker_step_data'


def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    model_list = [
        os.path.join(file_dir, 'step_policy/td3_step_model.zip'),
        os.path.join(file_dir, 'step_policy/td3_step_model.zip'),
        os.path.join(file_dir, 'step_policy/td3_step_model.zip'),
        os.path.join(file_dir, 'step_policy/td3_step_model.zip')
    ]

    step_sizes = [0.1 + i * 0.1 for i in range(0, 12)]  # 0.1 to 1.2, step: 0.1
    speeds = [0.005 + 0.013 * i for i in range(0, 16)]  # 0.005 to 0.2, step: 0.013

    from data.gen.walker_step.walker_step_rollout import Rollout_Generator
    rollout_generator = Rollout_Generator(model_paths=model_list)
    experiment_records = rollout_generator.gen_rollouts(step_sizes, speeds)
    print('-' * 20)
    print(f'[INFO] obtain {len(experiment_records)} successful records.')
    record_speed = [r['traj_meta']['softness_speed'] for r in experiment_records]
    record_step_size = [r['traj_meta']['final_step_size'] for r in experiment_records]
    print(f'[INFO] min step: {np.min(record_step_size)} | max step: {np.max(record_step_size)}')
    print(f'[INFO] min speed: {np.min(record_speed)} | max speed: {np.max(record_speed)}')

    video_path = f'{DATA_SAVE_DIR}/'
    os.makedirs(video_path, exist_ok=True)
    with open(f'{DATA_SAVE_DIR}/training_data.pickle', 'wb') as handle:
        pickle.dump(experiment_records, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
