import pickle
import sys

import numpy as np
import torch
from addict import Dict
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import get_attr_rep
from utils.traj_utils import pad_traj


class Attribute_Data(Dataset):
    def __init__(self, data_path, attr_info, is_video_data,
                 use_language_attr, encoder_path=None,
                 frame_stack=3, image_size=64,
                 max_data_cnt=sys.maxsize, subset_selector=None, **kwargs):
        """
        Read behavior data
        - data_path: path to the data
        - is_video_data: whether is image-based representation or state-based representation
        - use_language_attr: whether to use language embedding to represent attributes
        - encoder_path: path to the trained encoder (only needed if using image-based representation)
        - frame_stack: num of frame stack in an MDP state
        - image_size (only for video dataset): the rescaled frame size
        - max_data_cnt: can be used in testing to limit the number of data to read
        - subset_selector: a dict that can select subsets of data to use
        """
        self.data_path = data_path
        self.attr_info = attr_info
        self.attr_rep_method = 'language' if use_language_attr else 'one-hot'
        if is_video_data:
            from data.behavior_data import Video_Data
            self.behavior_data_cls = Video_Data
        else:
            from data.behavior_data import States_Data
            self.behavior_data_cls = States_Data
        self.encoder_path = encoder_path
        self.frame_stack = frame_stack
        self.max_data_cnt = max_data_cnt
        self.image_size = image_size
        self.subset_selector = subset_selector
        self.kwargs = kwargs
        # any class that inherits this class should maintain the following information
        self.encoder = None
        self.encoder_cfg = None
        self.behavior_data = None
        self.attributes = None      # attribute info
        self.is_global_ranking = None
        self.id2attr = None
        self.n_attr = 0
        self.attr_data = None
        self.max_traj_len = None
        self.traj_masks = None
        # init behavior data
        self._init_encoder()
        self._init_behavior_data()
        self._init_traj_mask()
        self._set_attribute_info()
        # sample subsets and init data
        self.trajs_subset = set()
        self.sample_subset(self.subset_selector)
        self.init_attribute_data()

    def _set_attribute_info(self):
        self.is_global_ranking = self.attr_info.use_global_ranking
        from omegaconf import OmegaConf
        self.attributes = Dict(OmegaConf.to_object(self.attr_info.attributes))
        self.id2attr = {self.attributes[attr].id: attr for attr in self.attributes}
        self.n_attr = len(self.id2attr)

    def _init_encoder(self):
        if self.encoder_path is not None:
            from encoders import get_trained_encoder
            self.encoder, _, self.encoder_cfg = get_trained_encoder(load_from=self.encoder_path,
                                                                    default_cfg=None, return_optimizer=False)

    def _init_behavior_data(self):
        behavior_data_cls = self.behavior_data_cls
        self.behavior_data = behavior_data_cls(self.data_path, self.frame_stack,
                                               0, self.encoder, self.image_size,
                                               self.max_data_cnt, **self.kwargs)

    def _init_traj_mask(self):
        self.max_traj_len = np.max([len(traj) for traj in self.behavior_data.traj_states])
        # with max_traj_len, we can generate a set of traj masks
        self.traj_masks = list()
        for traj_idx in range(len(self.behavior_data.traj_states)):
            traj_len = len(self.behavior_data.traj_states[traj_idx])
            traj_mask = [1 for _ in range(traj_len)] + [0 for _ in range(self.max_traj_len - traj_len)]
            self.traj_masks.append(np.array(traj_mask))

    @property
    def n_trajectories(self):
        return len(self.behavior_data.traj_meta_info)

    def resample_subset(self, subset_selector=None):
        self.sample_subset(subset_selector)
        self.init_attribute_data()

    def sample_subset(self, subset_selector=None):
        if subset_selector is None:
            self.trajs_subset = set([idx for idx in range(self.n_trajectories)])
            return
        trajs_subset = list()
        if 'attr_subset' in subset_selector and subset_selector['attr_subset'] is not None:
            traj_meta_info = self.behavior_data.traj_meta_info
            attr_subset = OmegaConf.to_object(subset_selector['attr_subset'])
            for traj_idx in range(self.n_trajectories):
                is_selected = True
                for attr_selection in attr_subset:
                    attr_selection = Dict(attr_selection)
                    attr_key = attr_selection.attr_key
                    ground_attr_score = traj_meta_info[traj_idx][attr_key]
                    lower_bound, upper_bound = attr_selection.lower, attr_selection.upper
                    if lower_bound is None and upper_bound is not None:
                        if ground_attr_score > upper_bound:
                            is_selected = False
                            break
                    elif lower_bound is not None and upper_bound is None:
                        if ground_attr_score < lower_bound:
                            is_selected = False
                            break
                    elif upper_bound is not None and lower_bound is not None:
                        if lower_bound > ground_attr_score or ground_attr_score > upper_bound:
                            is_selected = False
                            break
                if is_selected:
                    trajs_subset.append(traj_idx)
        else:
            trajs_subset = [idx for idx in range(len(self.behavior_data.traj_meta_info))]

        if 'random_subset' in subset_selector:
            subset_size = subset_selector['random_subset']
            assert subset_size <= len(trajs_subset)
            # randomly pick from the trajs_subset
            trajs_subset = list(np.random.choice(trajs_subset, subset_size, replace=False))
        self.trajs_subset = set(trajs_subset)
        assert len(self.trajs_subset) == len(trajs_subset)
        print(f'[INFO] selected {len(self.trajs_subset)} trajectories as data subset.')

    def init_attribute_data(self):
        self.attr_data = list()
        for traj_idx_0 in tqdm(range(self.n_trajectories), desc='Extracting attribute labels...'):
            if traj_idx_0 not in self.trajs_subset:
                continue
            for traj_idx_1 in range(traj_idx_0 + 1, len(self.behavior_data.traj_meta_info)):
                if traj_idx_1 not in self.trajs_subset:
                    continue
                for attr_name in self.attributes:
                    label = self._get_global_ranking(attr_name, traj_idx_0, traj_idx_1)
                    if label is not None:
                        attr_pair = (attr_name, self.attributes[attr_name].id, traj_idx_0, traj_idx_1, label)
                        self.attr_data.append(attr_pair)
        print(f'[INFO] extracted {len(self.attr_data)} attribute labels.')

    def _get_global_ranking(self, attr_name, traj_0_idx, traj_1_idx):
        traj_meta_info = self.behavior_data.traj_meta_info
        attr_0 = traj_meta_info[traj_0_idx][self.attributes[attr_name].key]
        attr_1 = traj_meta_info[traj_1_idx][self.attributes[attr_name].key]
        label = 0.5
        if attr_1 > attr_0 + self.attributes[attr_name].epsilon:
            label = 1.0
        elif attr_0 > attr_1 + self.attributes[attr_name].epsilon:
            label = 0
        if self.attributes[attr_name].reverse:
            return 1.0 - label
        else:
            return label

    def __len__(self):
        return len(self.attr_data)

    def get(self, idx):
        attr_name, attr_id, traj_idx_0, traj_idx_1, label = self.attr_data[idx]
        traj_states_0 = pad_traj(self.behavior_data.traj_states[traj_idx_0], self.max_traj_len)
        traj_states_1 = pad_traj(self.behavior_data.traj_states[traj_idx_1], self.max_traj_len)
        # represent attr as an one-hot vector
        attr_rep = get_attr_rep(attr_name, self.attributes, self.attr_rep_method)
        # traj mask
        traj_mask_0, traj_mask_1 = self.traj_masks[traj_idx_0], self.traj_masks[traj_idx_1]
        return attr_rep, traj_states_0, traj_mask_0, traj_states_1, traj_mask_1, label

    def __getitem__(self, idx):
        attr_rep, traj_states_0, traj_mask_0, traj_states_1, traj_mask_1, label = self.get(idx)

        torch_attr = torch.from_numpy(attr_rep).float()
        torch_traj_0, torch_traj_1 = torch.from_numpy(traj_states_0).float(), torch.from_numpy(traj_states_1).float()
        torch_mask_0, torch_mask_1 = torch.from_numpy(traj_mask_0).float(), torch.from_numpy(traj_mask_1).float()
        torch_label = torch.from_numpy(np.array([1.0 - label, float(label)])).float()
        weight_label = 1.0 if label == 0.5 else 1.0
        torch_weight_label = torch.tensor(weight_label)
        return torch_attr, torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, torch_label, torch_weight_label



