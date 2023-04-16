import os
import pickle

import numpy as np


DATA_SAVE_DIR = 'manipulator_lifting_data'


def main():
    n_repeat = 2
    experiment_records = list()
    # the rollout_generator will stochastically generate the behaviors
    forces = [0.35 + i * 0.05 for i in range(13)]     # 0.35 to 0.95, step size: 0.05
    hardness_degrees = [0.06 * i for i in range(16)]   # 0 to 0.9, step size: 0.06
    from data.gen.manipulator_lifting.lifting_rollout import Rollout_Generator
    rollout_generator = Rollout_Generator()
    expr_records = rollout_generator.gen_rollouts(forces=forces,
                                                  hardness_degrees=hardness_degrees,
                                                  n_repeat=n_repeat)
    experiment_records.extend(expr_records)

    print('-' * 20)
    print(f'[INFO] obtain {len(experiment_records)} successful records.')
    record_forces = [r['traj_meta']['force'] for r in experiment_records]
    record_hardness_degrees = [r['traj_meta']['hardness'] for r in experiment_records]
    print(f'[INFO] record_forces - min: {np.min(record_forces)} | max: {np.max(record_forces)}')
    print(f'[INFO] record_hardness_degrees - min: {np.min(record_hardness_degrees)} | max: {np.max(record_hardness_degrees)}')

    video_path = f'{DATA_SAVE_DIR}/'
    os.makedirs(video_path, exist_ok=True)
    with open(f'{DATA_SAVE_DIR}/test_data.pickle', 'wb') as handle:
        pickle.dump(experiment_records, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
