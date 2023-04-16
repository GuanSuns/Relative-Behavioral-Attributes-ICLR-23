import numpy as np
import torch
from addict import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.utils import get_attr_rep
from utils.logging_tools import Loss_Info


def eval_model(attr_data, attr_func, loss_func, cfg,
               pred_thresholds=(0.7, ), verbose=False,
               print_incorrect_pred=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attr_func_cfg = cfg.attr_func
    data_loader = DataLoader(attr_data,
                             num_workers=4,
                             batch_size=attr_func_cfg.eval_batch_size,
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

    attr_func.eval()
    with torch.no_grad():
        print('-' * 20)
        for i, data in enumerate(tqdm(data_loader, desc='Evaluating the model')):
            torch_attr, torch_traj_0, torch_mask_0, torch_traj_1, torch_mask_1, torch_label, _ = data
            torch_attr = torch_attr.to(device)
            torch_traj_0 = torch_traj_0.to(device)
            torch_mask_0 = torch_mask_0.to(device)
            torch_traj_1 = torch_traj_1.to(device)
            torch_mask_1 = torch_mask_1.to(device)
            torch_label = torch_label.to(device)

            pred = attr_func(torch_attr, torch_traj_0, torch_traj_1,
                             torch_mask_0, torch_mask_1)
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
            for j in range(batch_size):
                attr_name, attr_id, traj_idx_0, traj_idx_1, label = attr_data.attr_data[data_idx]

                for threshold in pred_thresholds:
                    binary_pred = np.zeros_like(prob_pred).astype(np.int8)
                    binary_pred[prob_pred > threshold] = 2
                    binary_pred[prob_pred < (1.0 - threshold)] = 1
                    # examine the incorrect prediction
                    if binary_pred[j, 0] != ground_truth[j, 0]:
                        if verbose and print_incorrect_pred:
                            print('-' * 20)
                            print(f'threshold: {threshold}')
                            print(f'attr name: {attr_name} | rep: {torch_attr[j].data.cpu().numpy()} | label: {label} | data idx: {data_idx}')
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
            acc = float(stats[attr_name][threshold].n_correct) / stats[attr_name][threshold].n_total
            stats[attr_name][threshold]['all_acc'] = acc
            eval_stats[f'eval/{attr_name}/{threshold}/all_acc'] = acc
            if verbose:
                print('-' * 20)
                print(f'attr: {attr_name} | threshold: {threshold} | acc: {acc}')
            for i in [0, 1]:
                acc = float(stats[attr_name][threshold][i].n_correct) / stats[attr_name][threshold][i].n_total
                stats[attr_name][threshold][f'{i}_acc'] = acc
                eval_stats[f'eval/{attr_name}/{threshold}/{i}_acc'] = acc
                if verbose:
                    print(f'pred: {i} | threshold: {threshold} | acc: {acc} ({stats[attr_name][threshold][i].n_correct}/{stats[attr_name][threshold][i].n_total})')
    eval_stats.update(renamed_eval_loss)
    return eval_stats


def compute_attr_stats(attr_data, attr_func, verbose=False,
                       rep='one-hot'):
    """
    Compute the statistics of each attribute
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    behavior_data = attr_data.behavior_data
    attribute_info = attr_data.attributes
    attr_func.eval()
    traj_attr_scores = list()
    for traj_idx in range(len(behavior_data.traj_states)):
        traj_states = behavior_data.traj_states[traj_idx]
        attr_scores = Dict()
        for attr_name in attribute_info:
            attr_rep = get_attr_rep(attr_name, attribute_info, rep=rep)
            with torch.no_grad():
                torch_traj = torch.from_numpy(traj_states).float().to(device)
                torch_attr = torch.from_numpy(attr_rep).float().to(device)
                _, attr_score = attr_func.reward(torch_attr, torch_traj)
                attr_score = attr_score.detach().squeeze().cpu().numpy()
                attr_scores[attr_name]['raw'] = attr_score
                attr_scores[attr_name]['ground_truth'] = behavior_data.traj_meta_info[traj_idx][attribute_info[attr_name].key]
        traj_attr_scores.append(attr_scores)
    # compute attribute score stats
    attr_info = Dict()
    for attr_name in attribute_info:
        attr_info[attr_name]['max'] = float(np.max([attr_scores[attr_name]['raw'] for attr_scores in traj_attr_scores]))
        attr_info[attr_name]['min'] = float(np.min([attr_scores[attr_name]['raw'] for attr_scores in traj_attr_scores]))
        attr_info[attr_name]['ground_min'] = float(np.min([attr_scores[attr_name]['ground_truth'] for attr_scores in traj_attr_scores]))
        attr_info[attr_name]['ground_max'] = float(np.max([attr_scores[attr_name]['ground_truth'] for attr_scores in traj_attr_scores]))
        if verbose:
            print('-' * 20)
            print(f'attr name: {attr_name} | key: {attribute_info[attr_name].key}')
            print(attr_info[attr_name])
    return attr_info




