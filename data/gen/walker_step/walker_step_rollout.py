import itertools

import numpy as np
from addict import Dict
from tqdm import tqdm


class Rollout_Generator:
    def __init__(self, model_paths):
        from environments.walker.walker_step import Walker_Step_Env
        self.env = Walker_Step_Env(speed_early_stop=False)

        self.model_paths = model_paths
        self.models = list()
        # load models
        from stable_baselines3 import TD3
        for model_path in model_paths:
            model = TD3.load(model_path)
            self.models.append((model_path, model))

    def gen_rollouts(self, step_sizes, speeds):
        rollouts = list()
        rollout_cfg = list(itertools.product(self.models, step_sizes, speeds))
        print(f'[INFO] {len(rollout_cfg)} configs to run')
        for e in tqdm(rollout_cfg, desc='Generating rollouts ...'):
            step_size, speed = e[1], e[2]
            success, expr_name, rollout_record = self.gen_single_rollout(model=e[0],
                                                                         step_size=step_size,
                                                                         speed=speed)
            if success:
                rollouts.append(rollout_record)
        print(f'[INFO] generated {len(rollouts)} successful rollouts.')
        return rollouts

    def gen_single_rollout(self, model, step_size, speed):
        model_path, policy_model = model[0], model[1]
        expr_name = f'{step_size}_{speed}'
        success = False

        self.env.set_target(step_size_range=(step_size, step_size + 0.036),
                            target_speed_range=(speed, speed + 0.0056),
                            ideal_height_range=(0.015, 0.015))
        state_sequence = []
        rgb_frames = []
        info_list = []

        policy_state = self.env.reset()
        obs = self.env.get_raw_obs()
        rgb_obs = self.env.get_frame_rgb()
        n_step = 0
        done = False
        final_step_size, softness_speed = -1, -1

        while not done and n_step < 100:
            action, _states = policy_model.predict(policy_state, deterministic=True)
            next_policy_state, reward, done, info = self.env.step(action)
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

            if done and info.done_message == 'success':
                if info.feet_dist is not None and info.softness_speed is not None:
                    final_step_size = info.feet_dist
                    softness_speed = info.softness_speed
                    success = True
                else:
                    print(f'final_step_size ({final_step_size}) or softness_speed ({softness_speed}) is None.')
                    success = False
            n_step += 1

            obs = next_obs
            policy_state = next_policy_state
            rgb_obs = next_rgb

        expr_record = Dict({
            'ground_truth_attr': {
                'step_size': final_step_size,
                'softness': softness_speed
            },
            'rgb_frames': rgb_frames,
            'state_sequence': np.array(state_sequence),
            'info_list': info_list,
            'traj_meta': {
                'model_path': model_path,
                'expr_name': expr_name,
                'target_step_size': step_size,
                'target_speed': speed,
                'final_step_size': final_step_size,
                'softness_speed': softness_speed,
                'success': success
            }
        })
        return success, expr_name, expr_record



