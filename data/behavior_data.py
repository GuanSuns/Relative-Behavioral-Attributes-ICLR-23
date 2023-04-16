import pickle
import sys

import cv2
import numpy as np
import torch.cuda
import torch.utils.data
from addict import Dict
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import resize_sequence


class Behavior_Data(Dataset):
    def __init__(self, data_path, frame_stack=3, n_augmentations=0,
                 encoder=None, image_size=64,
                 max_data_cnt=sys.maxsize, **kwargs):
        """
        Read behavior data
        - data_path: path to the data
        - frame_stack: num of frame stack in an MDP state
        - n_augmentations (only for video dataset): num of augmentation(s)
        - encoder (only for video dataset): the encoder to encode each frame into an embedding
        - image_size (only for video dataset): the rescaled frame size
        - max_data_cnt: can be used in testing to limit the number of data to read
        """
        self.data_path = data_path
        self.frame_stack = frame_stack
        self.n_augmentations = n_augmentations
        self.encoder = encoder
        self.max_data_cnt = max_data_cnt
        self.image_size = image_size
        # classes that inherit this class should maintain the following variables and information
        self.traj_meta_info = None  # a list of dict for per-traj meta info
        self.frame_infos = None  # nested list of per-frame info [[dict_info_0, dict_info_1...], ..., [dict_info_0, ...]]
        self.rgb_frames = None  # nested list of RGB observations of each traj [[rgb_0, rgb_1, ...], ..., [rgb_0, rgb_1, ...]]
        self.traj_states = None  # nested list of MDP states of each traj (e.g., stacked consecutive frames)
        self.state_idxs = None  # map an index to a dict {'traj_idx': int, 'timestep': int, 'is_augmented_state': bool, 'aug_idx': int}

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class Video_Data(Behavior_Data):
    def __init__(self, data_path, frame_stack=3, n_augmentations=0,
                 encoder=None, image_size=64,
                 max_data_cnt=sys.maxsize, **kwargs):
        super().__init__(data_path, frame_stack, n_augmentations, encoder, image_size, max_data_cnt, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._read_and_process_data()

    def _read_and_process_data(self):
        """
        The data is a list of dictionary with the following keys:
            - traj_meta: some trajectory-level meta information
            - info_list: a list of dict that contains per-frame information
            - obs_sequence: a list of frames (observations)
        """
        print(f'\n[INFO] reading data from disk ...')
        with open(self.data_path, 'rb') as handle:
            behavior_records = pickle.load(handle)
            behavior_records = behavior_records[:min(len(behavior_records), self.max_data_cnt)]
        self.traj_meta_info = [Dict(r['traj_meta']) for r in behavior_records]
        self.frame_infos = [r['info_list'] for r in behavior_records]
        self.rgb_frames = [np.array(r['obs_sequence']).astype(np.uint8) for r in behavior_records]
        # dataset information
        traj_lens = [len(v) for v in self.rgb_frames]
        print(f'[INFO] read {len(behavior_records)} trajectories | '
              f'max traj len: {np.max(traj_lens)} | min traj len: {np.min(traj_lens)}\n')

        self.traj_states = list()
        self.state_idxs = dict()
        state_idx = 0
        # process each traj and convert raw frames to states
        for traj_idx in tqdm(range(len(self.rgb_frames)), desc='Process the dataset'):
            # resize image
            resized_rgb_frames = resize_sequence(self.rgb_frames[traj_idx], self.image_size)
            # convert to grayscale
            grayscale_frames = self._frames_to_grayscale(resized_rgb_frames)
            # stack consecutive frames (we don't normalize stacked grayscale frames here to save memory usage)
            stacked_frames = self._stack_sequence(grayscale_frames, self.frame_stack)
            states = self._encode_states(stacked_frames) if self.encoder is not None else stacked_frames
            self.traj_states.append(states)
            for t in range(len(states)):
                self.state_idxs[state_idx] = Dict(
                    {'traj_idx': traj_idx, 'timestep': t, 'is_augmented_state': False, 'aug_idx': None})
                state_idx += 1

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
        assert len(encoded_states) == len(states)
        return np.array(encoded_states)

    def _stack_sequence(self, frames, frame_stack):
        padded_traj = self._pad_sequence(frames, frame_stack)
        stacked_frames = list()
        for start_idx in range(0, len(padded_traj) - frame_stack + 1):
            stacked_frames.append(np.stack([padded_traj[start_idx + i] for i in range(frame_stack)], axis=0))
        assert len(stacked_frames) == len(frames), f'{len(stacked_frames)} not equal to {len(frames)}'
        return np.array(stacked_frames).astype(np.uint8)

    @staticmethod
    def _pad_sequence(gray_frames, frame_stack):
        """
        Add (frame_stack - 1) frames (np.zeros) to current trajectory
        """
        frame_shape = gray_frames[0].shape
        return np.concatenate(
            (np.zeros(shape=(frame_stack - 1, frame_shape[0], frame_shape[1])).astype(np.uint8), gray_frames), axis=0)

    @staticmethod
    def _frames_to_grayscale(frames):
        grayscale_frames = []
        for frame_idx in range(len(frames)):
            gray_frame = cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2GRAY)
            grayscale_frames.append(gray_frame)
        return np.array(grayscale_frames).astype(np.uint8)

    def __len__(self):
        return len(self.state_idxs)

    def get(self, idx):
        traj_idx, timestep = self.state_idxs[idx]['traj_idx'], self.state_idxs[idx]['timestep']
        state = self.traj_states[traj_idx][timestep]
        if len(state.shape) == 3:
            state = state / 255.0
        traj_info, state_info, rgb_frame = self.traj_meta_info[traj_idx], self.frame_infos[traj_idx][timestep], \
                                           self.rgb_frames[traj_idx][timestep]
        return state, traj_info, state_info, rgb_frame

    def __getitem__(self, idx):
        state = self.get(idx)[0]
        return torch.from_numpy(state).float()


class States_Data(Behavior_Data):
    def __init__(self, data_path, frame_stack=None, n_augmentations=0,
                 encoder=None, image_size=None,
                 max_data_cnt=sys.maxsize, **kwargs):
        super().__init__(data_path, frame_stack, n_augmentations, encoder, image_size, max_data_cnt, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._read_and_process_data()

    def _read_and_process_data(self):
        """
        The data is a list of dictionary with the following keys:
            - traj_meta: some trajectory-level meta information
            - info_list: a list of dict that contains per-frame information
            - obs_sequence or state_sequence: a list of states
            - rgb_frames (optional): a list of rgb frames
        """
        print(f'\n[INFO] reading data from disk ...')
        with open(self.data_path, 'rb') as handle:
            behavior_records = pickle.load(handle)
            behavior_records = behavior_records[:min(len(behavior_records), self.max_data_cnt)]
        self.traj_meta_info = [Dict(r['traj_meta']) for r in behavior_records]
        self.frame_infos = [r['info_list'] for r in behavior_records]
        if 'rgb_frames' in behavior_records[0]:
            self.rgb_frames = [np.array(r['rgb_frames']).astype(np.uint8) for r in behavior_records]

        self.traj_states = list()
        self.state_idxs = dict()
        state_idx = 0
        # process each traj and convert raw frames to states
        for traj_idx in tqdm(range(len(behavior_records)), desc='Process the dataset'):
            if 'obs_sequence' in behavior_records[traj_idx]:
                states = np.array(behavior_records[traj_idx]['obs_sequence'])
            else:
                states = np.array(behavior_records[traj_idx]['state_sequence'])
            self.traj_states.append(states)
            for t in range(len(states)):
                self.state_idxs[state_idx] = Dict(
                    {'traj_idx': traj_idx, 'timestep': t, 'is_augmented_state': False, 'aug_idx': None})
                state_idx += 1
        # dataset information
        traj_lens = [len(v) for v in self.traj_states]
        print(f'[INFO] read {len(behavior_records)} trajectories | '
              f'max traj len: {np.max(traj_lens)} | min traj len: {np.min(traj_lens)}\n')

    def __len__(self):
        return len(self.state_idxs)

    def get(self, idx):
        traj_idx, timestep = self.state_idxs[idx]['traj_idx'], self.state_idxs[idx]['timestep']
        state = self.traj_states[traj_idx][timestep]
        if len(state.shape) == 3:
            state = state / 255.0
        traj_info, state_info, rgb_frame = self.traj_meta_info[traj_idx], self.frame_infos[traj_idx][timestep], \
                                           self.rgb_frames[traj_idx][timestep]
        return state, traj_info, state_info, rgb_frame

    def __getitem__(self, idx):
        state = self.get(idx)[0]
        return torch.from_numpy(state).float()


def main():
    behavior_data = Video_Data(data_path='behavior_data/lane_change/training_data.pickle', max_data=20)
    print('dataset len: ', len(behavior_data))
    for idx in range(len(behavior_data)):
        state, traj_info, state_info, rgb_frame = behavior_data.get(idx)
        print(state.shape, traj_info, state_info, rgb_frame.shape,
              '\nstate.shape, traj_info, state_info, rgb_frame.shape')
        cv2.imshow('rgb', rgb_frame)
        cv2.imshow('stacked frame 0', (state[0, :] * 255.0).astype(np.uint8))
        cv2.imshow('stacked frame 1', (state[1, :] * 255.0).astype(np.uint8))
        cv2.imshow('stacked frame 2', (state[2, :] * 255.0).astype(np.uint8))
        key = cv2.waitKey(0)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
