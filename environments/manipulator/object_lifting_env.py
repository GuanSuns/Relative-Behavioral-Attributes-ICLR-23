from collections import deque
from pprint import pprint

import cv2
from gym import Env
from gym import spaces
import numpy as np
from environments.manipulator.env_loader import load_manipulator


class Lifting_Env(Env):
    def __init__(self, history_len=5):
        super(Lifting_Env, self).__init__()
        self.target_height = 0.75
        task = 'insert_ball'
        self.env = load_manipulator(task_name=task, task_kwargs={'target_pos': (-0.45, 0.45)})
        # info
        obs_spec = self.env.observation_spec()
        self.state_dim = obs_spec['arm_pos'].shape[0] * obs_spec['arm_pos'].shape[1]
        self.history_len = history_len
        self.curr_obs = deque(maxlen=history_len)
        self.total_steps = 0
        self.reset()

    def reset(self, **kwargs):
        # reset obs
        self.curr_obs = deque(maxlen=self.history_len)
        [self.curr_obs.append(np.array([0 for _ in range(self.state_dim)])) for _ in range(self.history_len)]
        timestep = self.env.reset()
        arm_state = self._get_arm_state(timestep)
        self.curr_obs.append(np.copy(arm_state))
        return self.get_raw_obs()

    @property
    def action_space(self):
        action_dim = 5
        act_space = spaces.Box(low=np.array([-1.0 for _ in range(action_dim)]),
                               high=np.array([1.0 for _ in range(action_dim)]),
                               dtype=np.float32)
        return act_space

    @property
    def observation_space(self):
        obs_space = spaces.Box(low=-float('inf'),
                               high=float('inf'),
                               shape=(self.history_len * self.state_dim,),
                               dtype=np.float32)
        return obs_space

    @property
    def physics(self):
        return self.env.physics

    @staticmethod
    def _get_arm_state(timestep):
        obs = timestep.observation
        return np.reshape(obs['arm_pos'], -1)

    def step(self, action):
        self.total_steps += 1
        timestep = self.env.step(action)
        arm_state = self._get_arm_state(timestep)
        self.curr_obs.append(np.copy(arm_state))

        done = False
        success = False
        if self.physics.named.data.xpos['hand'][2] > self.target_height:
            done = True
            success = True
        # compute rewards
        reward = 0
        if not done:
            reward = self.physics.named.data.xpos['hand'][2] - self.target_height
        done = done or timestep.last()
        # info
        info = {
            'success': success,
            'reward': reward,
            'original_rew': timestep.reward,
            '_hand_pos': self.physics.named.data.xpos['hand']
        }
        return self.get_raw_obs(), reward, done, info

    def close(self):
        return self.env.close()

    def get_raw_obs(self):
        obs = np.concatenate(self.curr_obs, axis=None)
        return obs

    def get_frame_rgb(self, height=480, width=480):
        return self.env.physics.render(camera_id=0, height=height, width=width)

    def render(self, **kwargs):
        return self.get_frame_rgb()


def main():
    env = Lifting_Env()
    print(f'Observation space: {env.observation_space}.')
    print(f'Action space: {env.action_space}.')
    obs = env.reset()
    print(f'obs.shape: ', obs.shape)

    done = False
    n_step = 0
    while not done:
        n_step += 1
        img_obs = env.render()
        img_obs = cv2.cvtColor(img_obs, cv2.COLOR_RGB2BGR)
        cv2.imshow('RGB', img_obs)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        action = env.action_space.sample()
        _, _, done, info = env.step(action)
        print(f'[INFO] {n_step}')
        pprint(info)


if __name__ == '__main__':
    main()
