from dm_control import suite
from gym import Env
from gym import spaces
import numpy as np
import os
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
import cv2

from environments.walker.env_loader import load_walker


class Walker_Step_Env(Env):
    def __init__(self, speed_early_stop=True, step_size_range=None,
                 midpoint_pos_range=None,
                 ideal_height_range=None, speed_range=None):
        super(Walker_Step_Env, self).__init__()
        task = 'stand'
        self.env = load_walker(task_name=task)
        self.speed_early_stop = speed_early_stop

        # load the stand policy
        self.stand_policy = None
        self.stand_model_path = None
        self._init_stand_policy()

        # info
        self.init_foot_pos = {'right': [], 'left': []}
        self._last_observation = None
        self._last_feet_pos = None
        self._right_foot_velocity = 0
        self._midpoint_speed = None
        self._landing_speed = None

        # task config
        self.ideal_height_range = (0.01, 0.05) if ideal_height_range is None else ideal_height_range
        self.midpoint_pos_range = (0.5, 0.5) if midpoint_pos_range is None else midpoint_pos_range
        self.step_size_range = (0.1, 1.2) if step_size_range is None else step_size_range
        self.target_speed_range = (0.005, 0.2) if speed_range is None else speed_range

        self.n_timestep = 0
        self.curr_timestep = None
        self.max_timestep = 500
        # targets
        self.midpoint_pos = 0.5
        self.step_size = 0.6
        self.ideal_height = 0.01
        self.max_speed = 0.1
        self.target_speed = self.max_speed
        self.target_normalized_speed = 1.0

        self.left_foot_threshold = 0.05
        self.right_foot_threshold = 0.05
        self.final_standing_threshold = 0.85

        # execute stand policy
        self.is_standing_init = False
        self.stand_rew_threshold = 0.996
        self.execute_stand_policy()

        # these should be init in the reset func
        self.target_pos = None
        self.target_midpoint = None

    def set_target(self, ideal_height_range=None,
                   midpoint_pos_range=None,
                   step_size_range=None,
                   target_speed_range=None):
        """
        Target must be specified before calling reset
        """
        self.ideal_height_range = self.ideal_height_range if ideal_height_range is None else ideal_height_range
        self.midpoint_pos_range = self.midpoint_pos_range if midpoint_pos_range is None else midpoint_pos_range
        self.step_size_range = self.step_size_range if step_size_range is None else step_size_range
        self.target_speed_range = self.target_speed_range if target_speed_range is None else target_speed_range

    def _init_stand_policy(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.stand_model_path = os.path.join(file_dir, 'td3_stand_model.zip')
        self.stand_policy = TD3.load(self.stand_model_path)

    def execute_stand_policy(self):
        timestep = self.env.reset()
        obs = self.get_observation(timestep, for_standing=True)
        rewards = []
        n_step = 0
        while True and n_step < 200:
            action, _states = self.stand_policy.predict(obs)
            timestep = self.env.step(action)
            obs = self.get_observation(timestep, for_standing=True)
            rewards.append(timestep.reward)
            n_step += 1
            if (sum(rewards[-5:]) / 5) > self.stand_rew_threshold:
                break
        return obs, rewards[-1], timestep

    @property
    def action_space(self):
        act_spec = self.env.action_spec()
        act_space = spaces.Box(low=act_spec.minimum,
                               high=act_spec.maximum,
                               dtype=np.float32)
        # print(act_space)
        return act_space

    @property
    def observation_space(self):
        # Position and velocity
        obs_spec = self.env.observation_spec()
        obs_space = spaces.Box(low=-float('inf'),
                               high=float('inf'),
                               shape=(obs_spec['orientations'].shape[0] + obs_spec['velocity'].shape[0] + 1 + 4,),
                               dtype=np.float32)
        return obs_space

    @property
    def physics(self):
        return self.env.physics

    def get_observation(self, timestep, for_standing=False):
        if for_standing:
            obs = timestep.observation
            obs = np.concatenate((obs['orientations'], np.array([obs['height']]), obs['velocity']))
            return obs
        else:
            obs = timestep.observation
            obs = np.concatenate((obs['orientations'],
                                  np.array([obs['height']]),
                                  obs['velocity'],
                                  np.array([self.step_size, self.midpoint_pos, self.ideal_height, self.target_normalized_speed])))
            return obs

    def get_raw_obs(self):
        obs = self.curr_timestep.observation
        obs = np.concatenate((obs['orientations'], np.array([obs['height']]), obs['velocity']))
        return obs

    @staticmethod
    def _get_dist(pos_0, pos_1):
        if len(pos_0) > 2:
            pos_0 = np.array([pos_0[0], pos_0[2]])
        if len(pos_1) > 2:
            pos_1 = np.array([pos_1[0], pos_1[2]])
        return np.linalg.norm(np.array(pos_0) - np.array(pos_1))

    def get_reward(self, timestep):
        reward_info = dict()
        curr_left_pos = self.physics.named.data.geom_xpos['left_foot']
        curr_right_pos = self.physics.named.data.geom_xpos['right_foot']

        reward = 0
        # if the right foot hasn't reached the midpoint, encourage it to be closer to the target height
        if curr_right_pos[0] < self.target_midpoint[0]:
            midpoint_reward = np.clip(1.0 - self._get_dist(curr_right_pos, self.target_midpoint) / (self.step_size * self.midpoint_pos), 0, 1)
            # speed reward
            if self._right_foot_velocity <= 0:
                speed_reward = -1.0
            else:
                speed_reward = np.clip(1.0 - np.abs(self._right_foot_velocity - self.target_speed) / self.target_speed, -0.5, 1)
            reward_info['midpoint_reward'], reward_info['speed_reward'] = midpoint_reward, speed_reward
            reward = reward + speed_reward + midpoint_reward
        else:
            right_target_reward = np.clip(1.0 - self._get_dist(curr_right_pos, self.target_pos) / (self.step_size * self.midpoint_pos), 0, 1)
            reward_info['right_target_reward'] = right_target_reward
            reward += right_target_reward
        reward = reward / 5.0
        # encourage standing at the target pos
        if self._get_dist(curr_left_pos, self.init_foot_pos['left']) < self.left_foot_threshold:
            if self._get_dist(self.target_pos, curr_right_pos) < self.right_foot_threshold:
                reward = reward + 1.0 + timestep.reward
                if timestep.reward > self.final_standing_threshold:
                    reward += (self.max_timestep - self.n_timestep)
        return reward, reward_info

    def get_feet_dist(self, has_dir=False):
        right, left = self.physics.named.data.geom_xpos['right_foot'], self.physics.named.data.geom_xpos['left_foot']
        feet_dist = np.abs(left[0] - right[0]) if not has_dir else left[0] - right[0]
        return feet_dist

    def is_done(self, timestep):
        # end if the robot falls to the ground
        if timestep.reward < self.final_standing_threshold:
            return True, 'timestep.reward < self.final_standing_threshold'
        curr_left_pos, curr_right_pos = self.physics.named.data.geom_xpos['left_foot'], self.physics.named.data.geom_xpos['right_foot']
        # end if the left foot moves
        if self._get_dist(curr_left_pos, self.init_foot_pos['left']) > self.left_foot_threshold:
            return True, 'self._get_dist(curr_left_pos, self.init_foot_pos[\'left\']) > self.left_foot_threshold'
        # if the right foot moves to the right side of target
        if curr_right_pos[0] > self.target_pos[0] + 2 * self.right_foot_threshold:
            return True, 'curr_right_pos[0] > self.target_pos[0] + 2 * self.right_foot_threshold'
        if timestep.last() or self.n_timestep > self.max_timestep:
            return True, 'timestep.last() or self.n_timestep > self.max_timestep'
        # terminate if the right foot is moving in wrong direction or not in a reasonable range of the target speed
        if curr_right_pos[0] < self.target_midpoint[0]:
            if self._right_foot_velocity < 0:
                return True, 'self._right_foot_velocity < 0'
            if self.speed_early_stop:
                if curr_right_pos[0] > 0.7 * self.target_midpoint[0]:
                    if self._right_foot_velocity < 0.7 * self.target_speed or self._right_foot_velocity > 1.3 * self.target_speed:
                        return True, f'too fast or too slow, speed: {self._right_foot_velocity}, target: {self.target_speed}.'
        # terminate if success
        if self._get_dist(curr_left_pos, self.init_foot_pos['left']) < self.left_foot_threshold:
            if self._get_dist(self.target_pos, curr_right_pos) < self.right_foot_threshold:
                if timestep.reward > self.final_standing_threshold:
                    return True, 'success'
        return False, ''

    def get_right_foot_velocity(self):
        curr_right_pos = self.physics.named.data.geom_xpos['right_foot']
        last_right_pos = self._last_feet_pos['right']
        direction = 1.0 if curr_right_pos[0] - last_right_pos[0] > 0 else -1.0
        return direction * self._get_dist(curr_right_pos, last_right_pos)

    def step(self, action):
        self.n_timestep += 1
        timestep = self.env.step(action)
        self.curr_timestep = timestep
        obs = self.get_observation(timestep)

        self._last_observation = obs
        # update right foot speed before updating last_feet_pos
        curr_right_pos = self.physics.named.data.geom_xpos['right_foot']
        self._right_foot_velocity = self.get_right_foot_velocity()
        self._last_feet_pos = {'right': np.array(curr_right_pos),
                               'left': np.array(self.physics.named.data.geom_xpos['left_foot'])}
        if 0.7 * self.target_midpoint[0] < curr_right_pos[0] < self.target_midpoint[0]:
            self._midpoint_speed = self._right_foot_velocity if self._midpoint_speed is None else max(self._midpoint_speed, self._right_foot_velocity)
        if 0.7 * self.target_pos[0] < curr_right_pos[0] < 1.3 * self.target_pos[0]:
            right_foot_vel = np.abs(self._right_foot_velocity)
            self._landing_speed = right_foot_vel if self._landing_speed is None else max(self._landing_speed, right_foot_vel)

        done, done_message = self.is_done(timestep)
        curr_left_pos, curr_right_pos = self.physics.named.data.geom_xpos['left_foot'], \
                                        self.physics.named.data.geom_xpos['right_foot']

        softness_speed = None
        if self._midpoint_speed is not None and self._landing_speed is not None:
            softness_speed = max(self._midpoint_speed, self._landing_speed)
        info = {
            'target_speed': self.target_speed,
            'midpoint_speed': self._midpoint_speed,
            'landing_speed': self._landing_speed,
            'softness_speed': softness_speed,
            'right_foot_velocity': self._right_foot_velocity,
            'target_pos': self.target_pos[0],
            'right_foot_dist': self._get_dist(self.target_pos, curr_right_pos),
            'left_foot_dist': self._get_dist(curr_left_pos, self.init_foot_pos['left']),
            'standing_reward': timestep.reward,
            'feet_dist': self.get_feet_dist(has_dir=False),
            'done_message': done_message
        }
        reward, reward_info = self.get_reward(timestep)
        info.update(reward_info)

        # print(obs, self.env.physics.speed(), reward, done, info)
        return obs, reward, done, info

    def reset(self):
        # randomly sample task config
        self.step_size = np.random.uniform(self.step_size_range[0], self.step_size_range[1])
        self.ideal_height = np.random.uniform(self.ideal_height_range[0], self.ideal_height_range[1])
        self.midpoint_pos = np.random.uniform(self.midpoint_pos_range[0], self.midpoint_pos_range[1])
        self.target_speed = np.random.uniform(self.target_speed_range[0], self.target_speed_range[1])
        self.target_normalized_speed = self.target_speed / self.max_speed

        self.n_timestep = 0
        is_standing = False
        timestep = None
        while not is_standing:
            obs, init_standing_reward, timestep = self.execute_stand_policy()
            # check if the standing policy is successfully executed
            if self.get_feet_dist() < 0.06 and init_standing_reward > self.stand_rew_threshold:
                is_standing = True
            else:
                print('[INFO] walker-step env: re-attempt to initialize the env.')
        self._last_observation = self.get_observation(timestep)
        self.curr_timestep = timestep
        self._right_foot_velocity = 0
        self._midpoint_speed = None
        self._landing_speed = None
        self.init_foot_pos = {'right': np.array(self.physics.named.data.geom_xpos['right_foot']),
                              'left': np.array(self.physics.named.data.geom_xpos['left_foot'])}
        self._last_feet_pos = dict(self.init_foot_pos)
        self.target_pos = np.array([self.physics.named.data.geom_xpos['right_foot'][0] + self.step_size, 0])
        self.target_midpoint = np.array(
            [self.physics.named.data.geom_xpos['right_foot'][0] + self.step_size * self.midpoint_pos, self.ideal_height])

        return self._last_observation

    def close(self):
        return self.env.close()

    def render(self, mode="human"):
        return self.get_frame_rgb()

    def get_frame_rgb(self, ):
        return self.env.physics.render(camera_id=0, height=128, width=128)
