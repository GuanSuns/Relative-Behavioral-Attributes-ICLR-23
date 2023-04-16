import itertools

import cv2
import numpy as np
from addict import Dict
from tqdm import tqdm

EPSILON = 0.015


class Rollout_Generator:
    def __init__(self):
        from environments.manipulator.object_lifting_env import Lifting_Env
        self.env = Lifting_Env()

    def gen_rollouts(self, forces, hardness_degrees, n_repeat=2):
        rollouts = list()
        repeats = [i for i in range(n_repeat)]
        rollout_cfg = list(itertools.product(repeats, forces, hardness_degrees))
        print(f'[INFO] {len(rollout_cfg)} configs to run')
        for e in tqdm(rollout_cfg, desc='Generating rollouts ...'):
            force, hardness_degree = e[1], e[2]
            success, expr_name, rollout_record = self.gen_single_rollout(force=force, hardness=hardness_degree)
            if success:
                rollouts.append(rollout_record)
        print(f'[INFO] generated {len(rollouts)} successful rollouts.')
        return rollouts

    def gen_single_rollout(self, force, hardness):
        expr_name = f'{force}_{hardness}'

        state_sequence = []
        rgb_frames = []
        info_list = []

        self.env.reset()
        obs = self.env.get_raw_obs()
        rgb_obs = self.env.get_frame_rgb()
        n_step = 0
        done = False
        success = False

        reverse_force = -force
        while not done and n_step < 150:
            action = np.array([0 for _ in range(5)]).astype(np.float32)
            if n_step % 3 != 0:
                action[0] = force + np.random.uniform(0, EPSILON)
            else:
                action[0] = hardness * (reverse_force + np.random.uniform(0, EPSILON))
            action[1] = 0.02
            action[2] = np.random.uniform(-0.2, 0.2)
            action[-1] = 1.0

            _, reward, done, info = self.env.step(action)
            success = info['success']
            next_obs = self.env.get_raw_obs()
            next_rgb = self.env.get_frame_rgb()

            state_sequence.append(np.copy(obs))
            rgb_frames.append(np.copy(rgb_obs))
            info = Dict(info)
            info.update({'done': done,
                         'next_obs': np.copy(next_obs),
                         'reward': reward,
                         'action': action,
                         'timestep': n_step})
            info_list.append(info)

            n_step += 1
            obs = next_obs
            rgb_obs = next_rgb

        if not done or (done and not success):
            print(f'Unsuccessful rollout - force ({force}) or hardness ({hardness}).')

        expr_record = Dict({
            'ground_truth_attr': {
                'force': force,
                'hardness': hardness
            },
            'rgb_frames': rgb_frames,
            'state_sequence': np.array(state_sequence),
            'info_list': info_list,
            'traj_meta': {
                'expr_name': expr_name,
                'target_force': force,
                'target_hardness': hardness,
                'force': force,
                'hardness': hardness,
                'success': success
            }
        })
        return success, expr_name, expr_record



