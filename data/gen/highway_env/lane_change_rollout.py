import itertools

import gym
import torch
import numpy as np
import cv2
from addict import Dict
from tqdm import tqdm

from data.utils import resize_sequence
from environments.highway.lane_change_env import Lane_Change_Env


class Rollout_Generator:
    def __init__(self, encoder_path, image_size=64, frame_stack=3):
        # load the encoder
        import encoders
        self.encoder, _, _ = encoders.get_trained_encoder(encoder_path,
                                                          return_optimizer=False)
        self.image_size = image_size
        self.frame_stack = frame_stack

        self.env = gym.make("lanechange-v0", config={'manual_control': False})
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def gen_rollouts(self, speeds, turning_dists, tau_laterals, stochastic=False):
        rollouts = list()
        rollout_cfg = list(itertools.product(speeds, turning_dists, tau_laterals))
        for e in tqdm(rollout_cfg, desc='Generating rollouts ...'):
            turning_dist, tau_lateral = e[1], e[2]
            if stochastic:
                turning_dist = turning_dist + np.random.uniform(-0.1, 0.1)
                tau_lateral = tau_lateral + np.random.uniform(-0.01, 0.01)

            rollout_record = self.gen_single_rollout(speed=(e[0][0], e[0][1]),
                                                     turning_dist=turning_dist,
                                                     tau_lateral=tau_lateral)
            if rollout_record:
                rollouts.append(rollout_record)
        print(f'[INFO] generated {len(rollouts)} successful rollouts.')
        attr_stats = dict()
        env_attrs = rollouts[0].ground_truth_attr
        # noinspection PyUnresolvedReferences
        for attr in env_attrs:
            attr_stats[attr] = dict()
            attr_stats[attr]['min'] = np.min([r['ground_truth_attr'][attr] for r in rollouts])
            attr_stats[attr]['max'] = np.max([r['ground_truth_attr'][attr] for r in rollouts])
        print(f'[INFO] attribute ground truth info: {attr_stats}.')
        return rollouts

    def gen_single_rollout(self, speed, turning_dist, tau_lateral):
        def frames_to_grayscale(_frames):
            _grayscale_frames = []
            for _frame_idx in range(len(_frames)):
                _gray_frame = cv2.cvtColor(_frames[_frame_idx], cv2.COLOR_RGB2GRAY)
                _grayscale_frames.append(_gray_frame)
            return np.array(_grayscale_frames).astype(np.uint8)

        def pad_sequence(_gray_frames, _frame_stack):
            _frame_shape = _gray_frames[0].shape
            return np.concatenate(
                (np.zeros(shape=(_frame_stack - 1, _frame_shape[0], _frame_shape[1])).astype(np.uint8), _gray_frames),
                axis=0)

        def stack_sequence(_frames, _frame_stack):
            _padded_traj = pad_sequence(_frames, _frame_stack)
            _stacked_frames = list()
            for _start_idx in range(0, len(_padded_traj) - _frame_stack + 1):
                _stacked_frames.append(np.stack([_padded_traj[_start_idx + i] for i in range(_frame_stack)], axis=0))
            assert len(_stacked_frames) == len(_frames), f'{len(_stacked_frames)} not equal to {len(_frames)}'
            return np.array(_stacked_frames).astype(np.uint8)

        success = False
        final_turning_dist = -1.0
        final_turning_speed = -1.0
        final_other_vehicle_dist = -1.0

        # sample a trajectory
        obs = self.env.reset()
        step = 0
        done = False
        obs_sequence = list()
        info_list = list()

        # model-based policy
        from data.gen.highway_env.lane_change_synthetic_training import Policy
        policy = Policy(self.env, target_speed=speed, turning_dist=turning_dist, tau_lateral=tau_lateral)

        # noinspection PyUnresolvedReferences
        while not done and step < self.env.unwrapped.config["duration"]:
            action = policy.get_action()
            # save key info
            if action == policy.a_right and final_turning_speed < 0:
                final_turning_speed = policy.get_current_speed()
                final_turning_dist = policy.get_other_vehicle_dist()[0]
                success = True
            next_obs, reward, done, info = self.env.step(action)
            obs_sequence.append(np.copy(obs))
            info = Dict(info)
            info.update({'done': done, 'reward': reward, 'action': action,
                         'other_vehicle_dist': policy.get_other_vehicle_dist()[0],
                         'speed': policy.get_current_speed(), 'timestep': step})
            info_list.append(info)
            if info['crashed']:
                success = False

            if done:
                final_other_vehicle_dist = info['other_vehicle_dist']
            step += 1
            obs = next_obs

        if not success:
            return None

        # post process obs
        obs_sequence = np.array(obs_sequence).astype(np.uint8)
        resized_rgb_frames = resize_sequence(obs_sequence, self.image_size)
        # convert to grayscale
        grayscale_frames = frames_to_grayscale(resized_rgb_frames)
        stacked_frames = stack_sequence(grayscale_frames, self.frame_stack)
        states = self._encode_states(stacked_frames)

        rollout_record = Dict({
            'ground_truth_attr': {
                'sharpness': tau_lateral,
                'vehicle_dist': final_turning_dist
            },
            'rgb_frames': obs_sequence,
            'state_sequence': states,
            'info_list': info_list,
            'traj_meta': {
                'tau_lateral': tau_lateral,
                'turning_dist': turning_dist,
                'speed': speed,
                'final_turning_speed': final_turning_speed,
                'final_turning_dist': final_turning_dist,
                'final_other_vehicle_dist': final_other_vehicle_dist,
                'success': success
            }
        })
        return rollout_record

    def _encode_states(self, states):
        max_batch_size = int(64)
        n_states = len(states)
        encoded_states = list()
        # encode states
        self.encoder.eval()
        with torch.no_grad():
            for i in range(max(1, n_states // max_batch_size + 1)):
                batch_states = torch.from_numpy(
                    states[i * max_batch_size: min(n_states, (i + 1) * max_batch_size)]).float().to(self.device)
                batch_states = batch_states / 255.0
                state_embeddings = self.encoder.encode_and_sample_post(batch_states)[0].detach().cpu().numpy()
                encoded_states.extend([state_embeddings[i, :] for i in range(state_embeddings.shape[0])])
        assert len(encoded_states) == len(states), f'{len(encoded_states)}, {len(states)}'
        return np.array(encoded_states)
