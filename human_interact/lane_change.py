import argparse

import cv2
import imageio
import numpy as np
from addict import Dict


class Lane_Change_Interact:
    def __init__(self, agent, render=False, max_query=20):
        """
        The class that implements human-agent interaction
        """
        self.agent = agent
        self.target_attrs = None
        self.render = render
        self.max_query = max_query

    def set_target(self, target_attrs):
        self.target_attrs = target_attrs
        self.agent.reset()

    def gen_feedback(self, queried_behavior):
        attr_feedback = Dict()
        done = True
        for attr in self.target_attrs:
            curr_attr_score = queried_behavior.ground_truth_attr[attr]
            if curr_attr_score < self.target_attrs[attr].score - self.target_attrs[attr].epsilon:
                # the user wants to increase the strength
                attr_feedback[attr] = 1
            elif curr_attr_score > self.target_attrs[attr].score + self.target_attrs[attr].epsilon:
                # the user wants to decrease the strength
                attr_feedback[attr] = -1
            else:
                # the user is happy
                attr_feedback[attr] = 0
            done = done and (attr_feedback[attr] == 0)
        return attr_feedback, done

    @staticmethod
    def display_traj(rgb_sequence, n_query=0):
        end_session = False
        save_as_gif = False
        for rgb_obs in rgb_sequence:
            cv2.imshow('RGB obs', cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('e'):
                end_session = True
                break
            elif key == ord('s'):
                save_as_gif = True
        if save_as_gif:
            imageio.mimsave(f'lane_change_{n_query}.gif', rgb_sequence, duration=0.08)
        return end_session

    def start(self):
        n_query = 0
        n_attr_feedback = 0
        done = False
        end_session = False
        while not done and n_query < self.max_query and not end_session:
            n_query += 1
            print('-' * 20)
            print(f'[INFO] n query: {n_query}')
            queried_behavior = self.agent.get_query()
            queried_rgb_obs, queried_attr_score = queried_behavior.rgb_frames, queried_behavior.ground_truth_attr
            print(f'[INFO] #: {n_query} | queried attr score: {queried_attr_score} | target: {self.target_attrs}')
            attr_feedback, done = self.gen_feedback(queried_behavior)
            print(f'[INFO] #: {n_query} | feedback from human: {attr_feedback}.')
            for attr in attr_feedback:
                if attr_feedback[attr] != 0:
                    n_attr_feedback += 1

            if self.render:
                print(f'[INFO] trajectory len: {len(queried_rgb_obs)}')
                end_session = self.display_traj(queried_rgb_obs, n_query)
            self.agent.update_feedback(attr_feedback)
        print('#' * 10)
        if done:
            print(f'[INFO] successfully find target behavior with {n_query} queries.')
        else:
            print(f'[INFO] fail to find target behavior within {self.max_query} queries')
        return done, n_query, n_attr_feedback


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='method_2', type=str)
    parser.add_argument("--use-language", dest='use_language', action="store_true")
    parser.add_argument("--virtual", dest='virtual', action="store_true")
    sys_args = Dict()
    args, unknown = parser.parse_known_args()
    for arg in vars(args):
        sys_args[arg] = getattr(args, arg)
    return sys_args


def main():
    sys_args = get_args()
    if sys_args['virtual']:
        print('[INFO] setting up virtual display')
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1080, 608))
        display.start()

    algo = sys_args['algo']
    use_language = sys_args['use_language']

    target_attrs = Dict({
        'sharpness': {
            'score': [1.6066476961867366, 1.6082286685300085],  # min: 0.0579, max: 3.247
            'epsilon': 0.2,     # low precision: 0.2, high precision: 0.1
        },
        'vehicle_dist': {
            'score': [17.6263572297688, 30.23046737676121],  # min: 5.097, max: 30.3185
            'epsilon': 2.0,     # low precision: 3.0, high precision: 1.0
        }
    })
    attr_names = list(target_attrs.keys())

    if algo == 'method_1':
        from method_1.human_interact.lane_change_agent import Lane_Change_Agent
        if use_language:
            agent = Lane_Change_Agent(reward_func_path='trained_models/lane_change/language/method_1/reward_func')
        else:
            agent = Lane_Change_Agent(reward_func_path='trained_models/lane_change/method_1/reward_func')
    elif algo == 'method_2':
        from method_2.human_interact.lane_change_agent import Lane_Change_Agent
        if use_language:
            agent = Lane_Change_Agent(reward_func_path='trained_models/lane_change/language/method_2/')
        else:
            agent = Lane_Change_Agent(reward_func_path='trained_models/lane_change/method_2/')
    else:
        raise NotImplementedError

    human_interact = Lane_Change_Interact(agent, render=False)
    n_expr, n_success = len(target_attrs[attr_names[0]].score), 0
    n_queries, n_attr_feedbacks = list(), list()
    for expr_i in range(n_expr):
        expr_target = Dict(target_attrs)
        for attr_name in attr_names:
            expr_target[attr_name].score = target_attrs[attr_name].score[expr_i]
        print('-' * 20)
        print('-' * 20)
        print(f'[INFO] target attr scores: {expr_target}.')
        human_interact.set_target(expr_target)
        done, n_query, n_attr_feedback = human_interact.start()
        if done:
            n_attr_feedbacks.append(n_attr_feedback)
            n_queries.append(n_query)
            n_success += 1
    print('#' * 20)
    print('#' * 20)
    print(f'n expr: {n_expr} | n success: {n_success} | success rate: {float(n_success) / n_expr}')
    print(f'avg num of queries: {np.mean(n_queries)} | avg num of attr feedbacks: {np.mean(n_attr_feedbacks)}| std: {np.std(n_attr_feedbacks)}')


if __name__ == '__main__':
    main()