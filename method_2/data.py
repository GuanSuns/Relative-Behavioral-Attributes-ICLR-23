import os
import sys
from collections import defaultdict

import numpy as np
import torch
from addict import Dict
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import get_attr_rep
from utils.traj_utils import pad_traj


def get_attribute_data(cfg, read_test_data=False):
    # read attribute info
    domain_description = OmegaConf.load(cfg.domain_description_file)
    OmegaConf.update(cfg, 'attr_info',
                     domain_description.attr_info,
                     force_add=True)

    # determine whether to use language attr representation
    use_language_attr = False
    if 'use_language_attr' in cfg and cfg.use_language_attr:
        use_language_attr = True

    attr_training_data, attr_test_data = None, None
    attr_training_data = Attribute_Data(data_path=os.path.join(cfg.dataset_dir, 'training_data.pickle'),
                                        attr_info=cfg.attr_info,
                                        use_language_attr=use_language_attr,
                                        is_video_data=cfg.attr_info.is_video_data,
                                        encoder_path=cfg.reward_func.encoder_path,
                                        frame_stack=cfg.frame_stack,
                                        image_size=cfg.image_size,
                                        subset_selector=cfg.attr_training_subset,
                                        n_negative_samples=cfg.reward_func.n_negative_training,
                                        two_direction=cfg.reward_func.two_direction)
    if read_test_data:
        attr_test_data = Attribute_Data(data_path=os.path.join(cfg.dataset_dir, 'test_data.pickle'),
                                        attr_info=cfg.attr_info,
                                        use_language_attr=use_language_attr,
                                        is_video_data=cfg.attr_info.is_video_data,
                                        encoder_path=cfg.reward_func.encoder_path,
                                        frame_stack=cfg.frame_stack,
                                        image_size=cfg.image_size,
                                        subset_selector=cfg.attr_test_subset,
                                        n_negative_samples=cfg.reward_func.n_negative_training,
                                        two_direction=cfg.reward_func.two_direction)
    return attr_training_data, attr_test_data


class Attribute_Data(Dataset):
    def __init__(self, data_path, attr_info, is_video_data,
                 use_language_attr, encoder_path=None,
                 frame_stack=3, image_size=64,
                 max_data_cnt=sys.maxsize,
                 subset_selector=None, **kwargs):
        """
        This class is based on data.attribute_data.Attribute_Data
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
        self.n_negative_samples = kwargs['n_negative_samples']
        self.two_direction = kwargs['two_direction']
        self.subset_selector = subset_selector
        self.kwargs = kwargs
        # any class that inherits this class should maintain the following information
        self.encoder = None
        self.encoder_cfg = None
        self.behavior_data = None
        self.attributes = None  # attribute info
        self.is_global_ranking = None
        self.id2attr = None
        self.n_attr = 0
        self.attr_tuple_data = None
        self.attr_data = None
        self.max_traj_len = None
        self.traj_masks = None
        # init data
        self._init_encoder()
        self._init_behavior_data()
        self._init_traj_mask()
        self._set_attribute_info()
        # sample subsets and init data
        self.tuple_data_subset = set()
        self.sample_subset(self.subset_selector)
        self.init_attribute_data()

    def _set_attribute_info(self):
        from omegaconf import OmegaConf
        self.attributes = Dict(OmegaConf.to_object(self.attr_info.attributes))
        self.id2attr = {self.attributes[attr].id: attr for attr in self.attributes}
        self.n_attr = len(self.id2attr)
        # save attr tuple data
        self.attr_tuple_data = list()
        tuple_data_attr = defaultdict(lambda: 0)
        for traj_idx_0 in tqdm(range(self.n_trajectories), desc='Extracting attribute labels...'):
            for traj_idx_1 in range(traj_idx_0 + 1, self.n_trajectories):
                for attr_name in self.attributes:
                    is_increase = self._get_local_ranking(attr_name, traj_idx_0, traj_idx_1)
                    if is_increase is not None:
                        # whether to reverse label
                        label = is_increase
                        if self.attributes[attr_name].reverse:
                            label = 1.0 - label
                        # save labels
                        attr_tuple = (attr_name, self.attributes[attr_name].id, label, traj_idx_0, traj_idx_1)
                        self.attr_tuple_data.append(attr_tuple)
                        tuple_data_attr[attr_name] += 1
        print(f'[INFO] extracted {len(self.attr_tuple_data)} attr_tuple_data')
        print(f'[INFO] attr_tuple_data per attr: {dict(tuple_data_attr)}.')

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
            self.tuple_data_subset = set([idx for idx in range(len(self.attr_tuple_data))])
            return
        tuple_data_subset = list()
        if 'attr_subset' in subset_selector and subset_selector['attr_subset'] is not None:
            traj_meta_info = self.behavior_data.traj_meta_info
            attr_subset = OmegaConf.to_object(subset_selector['attr_subset'])
            for tuple_data_idx in range(len(self.attr_tuple_data)):
                _, _, _, traj_idx_0, traj_idx_1 = self.attr_tuple_data[tuple_data_idx]
                valid_tuple = True
                for traj_idx in [traj_idx_0, traj_idx_1]:
                    for attr_selection in attr_subset:
                        attr_selection = Dict(attr_selection)
                        attr_key = attr_selection.attr_key
                        ground_attr_score = traj_meta_info[traj_idx][attr_key]
                        lower_bound, upper_bound = attr_selection.lower, attr_selection.upper
                        if lower_bound is None and upper_bound is not None:
                            if ground_attr_score > upper_bound:
                                valid_tuple = False
                                break
                        elif lower_bound is not None and upper_bound is None:
                            if ground_attr_score < lower_bound:
                                valid_tuple = False
                                break
                        elif upper_bound is not None and lower_bound is not None:
                            if lower_bound > ground_attr_score or ground_attr_score > upper_bound:
                                valid_tuple = False
                                break
                    if not valid_tuple:
                        break
                if valid_tuple:
                    tuple_data_subset.append(tuple_data_idx)
        else:
            tuple_data_subset = set([idx for idx in range(len(self.attr_tuple_data))])

        if 'random_subset' in subset_selector:
            subset_size = subset_selector['random_subset']
            assert subset_size <= len(tuple_data_subset)
            # randomly pick from the trajs_subset
            tuple_data_subset = list(np.random.choice(list(tuple_data_subset), subset_size, replace=False))

        self.tuple_data_subset = set(tuple_data_subset)
        print(f'[INFO] selected {len(self.tuple_data_subset)} tuple data as data subset.')
        assert len(self.tuple_data_subset) == len(tuple_data_subset)

    def init_attribute_data(self):
        # append two-direction data
        attr_tuple_data = list()
        for tuple_data_idx in self.tuple_data_subset:
            attr_name, attr_id, label, traj_idx_0, traj_idx_1 = self.attr_tuple_data[tuple_data_idx]
            attr_tuple_data.append((attr_name, attr_id, label, traj_idx_0, traj_idx_1))
            if self.two_direction:
                attr_tuple = (attr_name, attr_id, 1.0 - label, traj_idx_1, traj_idx_0)
                attr_tuple_data.append(attr_tuple)
        # save attr_tuple_data into attr_data
        self.attr_data = list()
        for tuple_data_idx in range(len(attr_tuple_data)):
            attr_name, attr_id, is_increase, anchor_traj_idx, traj_idx_0 = attr_tuple_data[tuple_data_idx]
            self.attr_data.append((attr_name, attr_id, is_increase, anchor_traj_idx, anchor_traj_idx, traj_idx_0))
            for i in range(self.n_negative_samples):
                # use -1 to denote the idx of random negative samples (which will be replaced later when sampling)
                self.attr_data.append((attr_name, attr_id, is_increase, anchor_traj_idx, -1, traj_idx_0))
        # print info
        print(f'[INFO] total num of training samples: {len(self.attr_data)} | whether include two-direction data: {self.two_direction}')

    def _get_global_ranking(self, attr_name, traj_0_idx, traj_1_idx):
        traj_meta_info = self.behavior_data.traj_meta_info
        attr_0 = traj_meta_info[traj_0_idx][self.attributes[attr_name].key]
        attr_1 = traj_meta_info[traj_1_idx][self.attributes[attr_name].key]
        label = 0.5
        if attr_1 > attr_0 + self.attributes[attr_name].epsilon:
            label = 1.0
        elif attr_0 > attr_1 + self.attributes[attr_name].epsilon:
            label = 0
        return label

    def _is_locally_different(self, is_increase, attr_name, traj_0_idx, traj_1_idx):
        traj_meta_info = self.behavior_data.traj_meta_info
        attr_0 = traj_meta_info[traj_0_idx][self.attributes[attr_name].key]
        attr_1 = traj_meta_info[traj_1_idx][self.attributes[attr_name].key]
        if is_increase == 1.0:
            lower = attr_0 + self.attributes[attr_name].epsilon * self.attributes[attr_name].local_change_range.lower
            upper = attr_0 + self.attributes[attr_name].epsilon * self.attributes[attr_name].local_change_range.upper
            return lower < attr_1 < upper
        elif is_increase == 0:
            lower = attr_1 + self.attributes[attr_name].epsilon * self.attributes[attr_name].local_change_range.lower
            upper = attr_1 + self.attributes[attr_name].epsilon * self.attributes[attr_name].local_change_range.upper
            return lower < attr_0 < upper
        else:
            raise NotImplementedError

    def _get_local_ranking(self, attr_name, traj_0_idx, traj_1_idx, is_increase=None):
        # ensure other attributes are not significantly different
        for attr in self.attributes:
            if attr == attr_name:
                continue
            global_ranking = self._get_global_ranking(attr, traj_0_idx, traj_1_idx)
            if global_ranking != 0.5:
                return None
        # return None if the attr score is not significantly different
        local_label = self._get_global_ranking(attr_name, traj_0_idx, traj_1_idx)
        if local_label == 0.5:
            return None
        # ensure the two traces are only locally different and the change follows the direction of is_increase
        is_increase = is_increase if is_increase is not None else local_label
        if self._is_locally_different(is_increase, attr_name, traj_0_idx, traj_1_idx):
            return local_label
        # otherwise return None
        return None

    def __len__(self):
        return len(self.attr_data)

    @staticmethod
    def _random_exclude(n, exclude_list):
        while True:
            i = np.random.randint(n)
            if i not in exclude_list:
                return i

    def get(self, idx):
        attr_name, attr_id, is_increase, anchor_traj_idx, traj_idx_0, traj_idx_1 = self.attr_data[idx]
        if traj_idx_0 == -1:
            traj_idx_0 = self._random_exclude(len(self.behavior_data.traj_meta_info),
                                              exclude_list=[anchor_traj_idx, traj_idx_1])
        anchor_states = pad_traj(self.behavior_data.traj_states[anchor_traj_idx], self.max_traj_len)
        anchor_traj_len = self.behavior_data.traj_states[anchor_traj_idx].shape[0]
        traj_states_0 = pad_traj(self.behavior_data.traj_states[traj_idx_0], self.max_traj_len)
        traj_states_1 = pad_traj(self.behavior_data.traj_states[traj_idx_1], self.max_traj_len)
        # represent attr as an one-hot vector
        attr_rep = get_attr_rep(attr_name, self.attributes, self.attr_rep_method)
        # traj mask
        traj_mask_0, traj_mask_1 = self.traj_masks[traj_idx_0], self.traj_masks[traj_idx_1]
        return attr_rep, is_increase, anchor_states, anchor_traj_len, traj_states_0, traj_mask_0, traj_states_1, traj_mask_1, traj_idx_0

    def __getitem__(self, idx):
        attr_rep, is_increase, anchor_states, anchor_traj_len, traj_states_0, traj_mask_0, traj_states_1, traj_mask_1, traj_idx_0 = self.get(idx)
        torch_attr = torch.from_numpy(attr_rep).float()
        torch_is_increase = torch.tensor(is_increase).float().unsqueeze(-1)
        # traj_1 is always preferred over traj_0 given the anchor_traj and the direction is_increase
        torch_label = torch.from_numpy(np.array([0, 1.0])).float()
        torch_anchor_traj = torch.from_numpy(anchor_states).float()
        torch_traj_0, torch_traj_1 = torch.from_numpy(traj_states_0).float(), torch.from_numpy(traj_states_1).float()
        torch_mask_0, torch_mask_1 = torch.from_numpy(traj_mask_0).float(), torch.from_numpy(traj_mask_1).float()
        return torch_attr, torch_is_increase, torch_label, torch_anchor_traj, anchor_traj_len, torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, traj_idx_0

