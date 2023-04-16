import numpy as np
from addict import Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging_tools import Loss_Info


def eval_model(attr_data, reward_func, loss_func, cfg,
               pred_thresholds=(0.51, ), verbose=False, print_incorrect_pred=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reward_func_cfg = cfg.reward_func
    data_loader = DataLoader(attr_data,
                             num_workers=4,
                             batch_size=reward_func_cfg.eval_batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=False)
    data_idx = 0
    stats = Dict()
    stats_item = {
        'n_correct': 0,
        'n_total': 0,
        0: {'n_correct': 0,
            'n_total': 0, },
        1: {'n_correct': 0,
            'n_total': 0, },
    }
    stats['all_attr'] = Dict({threshold: Dict(stats_item) for threshold in pred_thresholds})
    eval_loss_info = Loss_Info()

    reward_func.eval()
    with torch.no_grad():
        print('-' * 20)
        for i, data in enumerate(tqdm(data_loader, desc='Evaluating the model')):
            torch_attr, torch_is_increase, torch_label, torch_anchor_traj, anchor_traj_len, torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, traj_0_idxs = data
            torch_attr = torch_attr.to(device)
            torch_is_increase = torch_is_increase.to(device)
            torch_label = torch_label.to(device)
            torch_anchor_traj = torch_anchor_traj.to(device)
            torch_traj_0 = torch_traj_0.to(device)
            torch_mask_0 = torch_mask_0.to(device)
            torch_traj_1 = torch_traj_1.to(device)
            torch_mask_1 = torch_mask_1.to(device)
            torch_traj_1 = torch_traj_1.to(device)
            torch_mask_1 = torch_mask_1.to(device)

            pred = reward_func(torch_attr, torch_is_increase, torch_anchor_traj, anchor_traj_len,
                               torch_traj_0, torch_traj_1, torch_mask_0, torch_mask_1)
            _, traj_reward_0, _, traj_reward_1 = pred
            pred_rewards = torch.cat((traj_reward_0, traj_reward_1), dim=-1)
            pred_loss = loss_func(pred_rewards, torch_label)
            loss_info = {
                'loss': pred_loss.data.cpu().numpy(),
                'avg_traj_reward': torch.mean(pred_rewards).data.cpu().numpy(),
            }
            eval_loss_info.update(loss_info)

            # compute the prediction accuracy
            # ground truth
            np_target_labels = np.expand_dims(torch_label.data.cpu().numpy()[:, 0], axis=-1)
            ground_truth = np.zeros_like(np_target_labels).astype(np.int8)
            ground_truth[np_target_labels > 0.55] = 2
            ground_truth[np_target_labels < 0.45] = 1

            # prediction
            np_trace_reward_0, np_trace_reward_1 = traj_reward_0.data.cpu().numpy(), traj_reward_1.data.cpu().numpy()
            prob_pred = 1.0 / (1 + np.exp(np_trace_reward_1 - np_trace_reward_0))

            batch_size = prob_pred.shape[0]
            traj_0_idxs = traj_0_idxs.cpu().numpy()
            for j in range(batch_size):
                attr_name, attr_id, is_increase, anchor_traj_idx, _, traj_idx_1 = attr_data.attr_data[data_idx]
                traj_idx_0 = traj_0_idxs[j]
                for threshold in pred_thresholds:
                    binary_pred = np.zeros_like(prob_pred).astype(np.int8)
                    binary_pred[prob_pred > threshold] = 2
                    binary_pred[prob_pred < (1.0 - threshold)] = 1
                    # examine the incorrect prediction
                    if binary_pred[j, 0] != ground_truth[j, 0]:
                        if verbose and print_incorrect_pred:
                            print('-' * 20)
                            print(f'threshold: {threshold}')
                            print(f'attr name: {attr_name} | rep: {torch_attr[j].data.cpu().numpy()} | is_increase: {is_increase} | data idx: {data_idx}')
                            anchor_traj_meta_info = attr_data.behavior_data.traj_meta_info[anchor_traj_idx]
                            print(f'anchor traj meta info: {anchor_traj_meta_info}')
                            print(f'traj 0 reward: {np_trace_reward_0[j, 0]} | traj 1 reward: {np_trace_reward_1[j, 0]} | prob: {prob_pred[j, 0]}')
                            traj_meta_info_0 = attr_data.behavior_data.traj_meta_info[traj_idx_0]
                            print(f'traj meta info 0: {traj_meta_info_0}')
                            traj_meta_info_1 = attr_data.behavior_data.traj_meta_info[traj_idx_1]
                            print(f'traj meta info 1: {traj_meta_info_1}')

                    if attr_name not in stats:
                        stats[attr_name] = Dict({threshold: Dict(stats_item) for threshold in pred_thresholds})

                    for _attr in [attr_name, 'all_attr']:
                        stats[_attr][threshold].n_total += 1
                        ground_label = 1 if ground_truth[j, 0] > 0 else 0
                        stats[_attr][threshold][ground_label].n_total += 1
                        if binary_pred[j, 0] == ground_truth[j, 0]:
                            stats[_attr][threshold].n_correct += 1
                            stats[_attr][threshold][ground_label].n_correct += 1
                data_idx += 1
    # rename loss info
    eval_loss_info = eval_loss_info.get_log_dict()
    renamed_eval_loss = Dict()
    for item in eval_loss_info:
        renamed_eval_loss[f'eval/{item}'] = eval_loss_info[item]

    eval_stats = Dict()
    for attr_name in list(stats):
        for threshold in pred_thresholds:
            acc = float(stats[attr_name][threshold].n_correct) / max(1, stats[attr_name][threshold].n_total)
            stats[attr_name][threshold]['all_acc'] = acc
            eval_stats[f'eval/{attr_name}/{threshold}/all_acc'] = acc
            if verbose:
                print('-' * 20)
                print(f'attr: {attr_name} | threshold: {threshold} | acc: {acc}')
            for i in [0, 1]:
                acc = float(stats[attr_name][threshold][i].n_correct) / max(1, stats[attr_name][threshold][i].n_total)
                stats[attr_name][threshold][f'{i}_acc'] = acc
                eval_stats[f'eval/{attr_name}/{threshold}/{i}_acc'] = acc
                if verbose:
                    print(f'pred: {i} | threshold: {threshold} | acc: {acc} ({stats[attr_name][threshold][i].n_correct}/{stats[attr_name][threshold][i].n_total})')
    eval_stats.update(renamed_eval_loss)
    return eval_stats
