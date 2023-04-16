import numpy as np
from numpy.linalg import norm
import torch
from addict import Dict
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import get_attr_rep
from method_1.reward_func.utils import to_attrs_vec
from utils.traj_utils import pad_traj


class Traj_Pref_Data(Dataset):
    # noinspection PyTypeChecker
    def __init__(self, attribute_data, n_targets_per_pair, cfg):
        self.cfg = cfg
        self.attr_rep_method = 'language' if ('use_language_attr' in cfg and cfg.use_language_attr) else 'one-hot'
        self.n_targets_per_pair = n_targets_per_pair
        self.attribute_data = attribute_data
        self.attributes = self.attribute_data.attributes
        self.behavior_data = self.attribute_data.behavior_data
        self.traj_meta_info = self.behavior_data.traj_meta_info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.attr_func_path = cfg.reward_func.attr_func_path
        self.attr_func = None
        self._init_attr_func()
        # encoded_attr_info is populated after attr_func is loaded
        self.encoded_attr_info = self.cfg.attr_func.encoded_attributes

        self.traj_attr_scores = None
        self.traj_attr_vector = None
        self._extract_attr_scores()

        self.data_idx_to_traj_pair = list()
        for traj_idx_0 in range(len(self.traj_attr_vector)):
            for traj_idx_1 in range(traj_idx_0 + 1, len(self.traj_attr_vector)):
                if self._is_significant_diff(traj_idx_0, traj_idx_1):
                    self.data_idx_to_traj_pair.append((traj_idx_0, traj_idx_1))

    def _is_significant_diff(self, traj_idx_0, traj_idx_1):
        for attr_name in self.attributes:
            attr_0 = self.traj_meta_info[traj_idx_0][self.attributes[attr_name].key]
            attr_1 = self.traj_meta_info[traj_idx_1][self.attributes[attr_name].key]
            if attr_1 > attr_0 + self.attributes[attr_name].epsilon:
                return True
            elif attr_0 > attr_1 + self.attributes[attr_name].epsilon:
                return True
        return False

    def _init_attr_func(self):
        from method_1.attr_func import get_trained_model
        self.attr_func, _, self.cfg = get_trained_model(self.attr_func_path, self.cfg, return_optimizer=False)

    def _extract_attr_scores(self):
        self.attr_func.eval()
        self.traj_attr_scores = list()
        print('-' * 20)
        for traj_idx in tqdm(range(len(self.behavior_data.traj_states)),
                             desc='Extracting raw attribute scores'):
            traj_states = self.behavior_data.traj_states[traj_idx]
            attr_scores = Dict()

            for attr_name in self.attributes:
                attr_rep = get_attr_rep(attr_name, self.attributes, rep=self.attr_rep_method)
                with torch.no_grad():
                    torch_traj = torch.from_numpy(traj_states).float().to(self.device)
                    torch_attr = torch.from_numpy(attr_rep).float().to(self.device)
                    _, attr_score = self.attr_func.reward(torch_attr, torch_traj)
                    attr_score = attr_score.detach().squeeze().cpu().numpy()
                attr_scores[attr_name] = attr_score
            self.traj_attr_scores.append(attr_scores)
        # save each trajectory's attribute vector
        attr_vectors = to_attrs_vec(self.traj_attr_scores, self.encoded_attr_info)
        assert len(attr_vectors) == len(self.traj_attr_scores)
        self.traj_attr_vector = list()
        for traj_idx in range(len(attr_vectors)):
            self.traj_attr_vector.append(attr_vectors[traj_idx, :])

    def _get_pref_label(self, traj_idx_0, traj_idx_1, target_attr_vector):
        attr_vector_0, attr_vector_1 = self.traj_attr_vector[traj_idx_0], self.traj_attr_vector[traj_idx_1]
        diff_0 = norm(attr_vector_0 - target_attr_vector)
        diff_1 = norm(attr_vector_1 - target_attr_vector)
        if diff_1 < diff_0:
            # traj 1 is preferred
            return 1
        else:
            # traj 0 is preferred
            return 0

    def get(self, idx):
        # convert data idx into traj idxs
        pair_idx = idx // int(self.n_targets_per_pair)
        traj_idx_0, traj_idx_1 = self.data_idx_to_traj_pair[pair_idx]
        # sample a target traj
        target_traj_idx = np.random.randint(len(self.traj_attr_vector))
        target_attr_vector = np.copy(self.traj_attr_vector[target_traj_idx])
        label = self._get_pref_label(traj_idx_0, traj_idx_1, target_attr_vector)
        # prepare data
        traj_states_0 = pad_traj(self.behavior_data.traj_states[traj_idx_0], self.attribute_data.max_traj_len)
        traj_states_1 = pad_traj(self.behavior_data.traj_states[traj_idx_1], self.attribute_data.max_traj_len)
        # traj mask
        traj_mask_0, traj_mask_1 = self.attribute_data.traj_masks[traj_idx_0], self.attribute_data.traj_masks[traj_idx_1]
        return traj_states_0, traj_mask_0, traj_states_1, traj_mask_1, target_attr_vector, label

    def __getitem__(self, idx):
        traj_states_0, traj_mask_0, traj_states_1, traj_mask_1, attr_vector_target, label = self.get(idx)
        torch_traj_0, torch_traj_1 = torch.from_numpy(traj_states_0).float(), torch.from_numpy(traj_states_1).float()
        torch_mask_0, torch_mask_1 = torch.from_numpy(traj_mask_0).float(), torch.from_numpy(traj_mask_1).float()
        torch_target_attr = torch.from_numpy(attr_vector_target).float()
        torch_label = torch.from_numpy(np.array([1.0 - label, float(label)])).float()
        return torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, torch_target_attr, torch_label

    def __len__(self):
        return len(self.data_idx_to_traj_pair) * self.n_targets_per_pair





