import itertools
import os
import pickle
import random
import shutil
import time

import gym
from addict import Dict
from tqdm import tqdm
import numpy as np

from environments.highway.lane_change_env import Lane_Change_Env

DATA_SAVE_DIR = 'lane_change_data'


class Policy:
    def __init__(self, env, target_speed=(7.5, 9.5), turning_dist=10, tau_lateral=0.1):
        self.env = env
        self.unwrapped_env = env.unwrapped
        self.controlled_vehicle = self.unwrapped_env.controlled_vehicles[0]
        self.vehicle_0 = self.unwrapped_env.other_vehicles[0]
        self.is_lane_changing = False
        self.sleep_time = 0
        # actions (doesn't have significant influence on the behavior)
        self.a_faster = 11  # [11, 14]
        self.a_slower = 15  # [15, 18]
        self.a_left = 0  # [0, 4]
        self.a_right = 6  # [6, 10]
        # config (will have significant influence on the behavior)
        self.target_speed = target_speed
        self.turning_dist = turning_dist
        self.controlled_vehicle.hard_code_param(is_on=True, params={'tau_lateral': tau_lateral})

    def get_current_speed(self):
        return self.controlled_vehicle.speed

    def get_other_vehicle_dist(self):
        dist_0 = self.controlled_vehicle.position[0] - self.vehicle_0.position[0]
        return [dist_0]

    def is_speed_in_range(self):
        speed = self.controlled_vehicle.speed
        return self.target_speed[0] < speed < self.target_speed[1]

    def get_action(self):
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        speed = self.controlled_vehicle.speed
        other_vehicle_dist = self.get_other_vehicle_dist()
        if not self.is_lane_changing:
            if other_vehicle_dist[0] > self.turning_dist and self.is_speed_in_range():
                self.is_lane_changing = True
                return self.a_right
            if speed > self.target_speed[1]:
                return self.a_slower
            if speed < self.target_speed[0]:
                return self.a_faster
        else:
            # finishing turning
            pass
        return None


def run_episode(target_speed=(7.5, 9.5), turning_dist=10, turning_dist_stochastic=0.1,
                tau_lateral=0.1, tau_lateral_stochastic=0.01, print_info=True, render=True):
    turning_dist = turning_dist + np.random.uniform(-turning_dist_stochastic, turning_dist_stochastic)
    tau_lateral = tau_lateral + np.random.uniform(-tau_lateral_stochastic, tau_lateral_stochastic)
    success = False
    final_turning_dist = -1.0
    final_turning_speed = -1.0
    final_other_vehicle_dist = -1.0
    obs_sequence = []
    info_list = []

    # new config
    config = {'manual_control': False}

    # make environment
    env = gym.make("lanechange-v0", config=config)
    if 'print_action_info' in dir(env.unwrapped.action_type) and print_info:
        env.unwrapped.action_type.print_action_info()
    # record video
    expr_name = f's_{target_speed[0]}_{target_speed[1]}_turn_dist_{turning_dist}_lateral_{tau_lateral}'
    # env = record_videos(env, path=video_path)
    obs, done = env.reset(), False

    # manual policy
    policy = Policy(env, target_speed=target_speed, turning_dist=turning_dist, tau_lateral=tau_lateral)

    # Run episode
    step = 0
    info = None
    while not done and step < env.unwrapped.config["duration"]:
        action = policy.get_action()
        # save key info
        if action == policy.a_right and final_turning_speed < 0:
            final_turning_speed = policy.get_current_speed()
            final_turning_dist = policy.get_other_vehicle_dist()[0]
            success = True
        next_obs, reward, done, info = env.step(action)
        obs_sequence.append(np.copy(obs))
        info = Dict(info)
        info.update({'done': done, 'reward': reward, 'action': action,
                     'other_vehicle_dist': policy.get_other_vehicle_dist()[0],
                     'speed': policy.get_current_speed(), 'timestep': step})
        info_list.append(info)

        if done:
            final_other_vehicle_dist = info['other_vehicle_dist']

        step += 1
        obs = next_obs
        if print_info:
            action_name = 'None'
            if 'get_name' in dir(env.unwrapped.action_type):
                action_name = env.unwrapped.action_type.get_name(action)
            print(f'[INFO] step {step}, action: {action_name}, info: {info}, reward: {reward}')
        if render:
            env.render()
    env.close()

    # remove files if crashed
    if info['crashed']:
        success = False
    expr_record = {
        'info_list': info_list,
        'obs_sequence': obs_sequence,
        'traj_meta': {
            'expr_name': expr_name,
            'tau_lateral': tau_lateral,
            'final_turning_speed': final_turning_speed,
            'final_turning_dist': final_turning_dist,
            'final_other_vehicle_dist': final_other_vehicle_dist,
            'success': success
        }
    }
    return success, expr_name, expr_record


def main():
    n_repeats = [i for i in range(1)]
    target_speeds = [(7.5, 8.5), (8.5, 9.5)]
    print(f'[INFO] n target speed: {len(target_speeds)}')
    turning_dists = [i for i in range(5, 31, 1)]
    print(f'[INFO] n turning dists: {len(turning_dists)}')
    tau_lateral = [0.1 + 0.2 * i for i in range(17)]
    print(f'[INFO] n tau lateral: {len(tau_lateral)}')

    experiments = list(itertools.product(n_repeats, target_speeds, turning_dists, tau_lateral))
    print(f'[INFO] {len(experiments)} experiments to run')

    experiment_records = []
    for e in tqdm(experiments):
        success, expr_name, expr_record = run_episode(target_speed=(e[1][0], e[1][1]), turning_dist=e[2],
                                                      tau_lateral=e[3], print_info=False, render=False)
        if success:
            experiment_records.append(expr_record)

    video_path = f'{DATA_SAVE_DIR}/'
    os.makedirs(video_path, exist_ok=True)
    with open(f'{DATA_SAVE_DIR}/training_data.pickle', 'wb') as handle:
        pickle.dump(experiment_records, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
