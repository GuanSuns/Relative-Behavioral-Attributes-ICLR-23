from dm_control import suite
from gym import Env
from gym import spaces
import numpy as np

from environments.walker.env_loader import load_walker


class Walker_Stand_Env(Env):
    """
    Default action space:

    Default observation space:
        joint angles - 5 internal joint angles
        vector of nose to target in local coordinate of the head
        velocity of the body - 6x3 velocities (x, y - linear & z - rotational)
    Reward:
        Distance from nose to target
    """

    def __init__(self):
        super(Walker_Stand_Env, self).__init__()
        task = 'stand'
        self.env = load_walker(task_name=task)
        # customize reward
        self.original_return = 0
        # constraint distance
        self.feet_dist_target = 0.05

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
                               shape=(obs_spec['orientations'].shape[0] + obs_spec['velocity'].shape[0] + 1,),
                               dtype=np.float32)
        return obs_space

    @property
    def physics(self):
        return self.env.physics

    @staticmethod
    def get_observation(timestep):
        obs = timestep.observation
        obs = np.concatenate((obs['orientations'], np.array([obs['height']]), obs['velocity']))
        return obs

    def get_feet_dist(self, has_dir=False):
        right, left = self.physics.named.data.geom_xpos['right_foot'], self.physics.named.data.geom_xpos['left_foot']
        feet_dist = np.abs(left[0] - right[0]) if not has_dir else left[0] - right[0]
        return feet_dist

    def step(self, action):
        timestep = self.env.step(action)
        obs = self.get_observation(timestep)

        standing_reward = timestep.reward
        self.original_return += standing_reward

        done = timestep.last()
        info = {'original_return': self.original_return,
                'feet_dist': self.get_feet_dist(has_dir=False)
                }
        # only apply feet dist reward when the agent is standing
        reward = standing_reward
        feet_dist = self.get_feet_dist()
        if standing_reward > 0.99:
            if feet_dist < self.feet_dist_target:
                reward += 1.0
            else:
                # the closer, the bigger reward
                dist_reward = 1.0 - np.clip(np.abs(self.feet_dist_target - feet_dist), 0, 1.0)
                reward += dist_reward
        return obs, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        obs = self.get_observation(timestep)
        self.original_return = 0
        return obs

    def close(self):
        return self.env.close()

    def render(self, mode="human"):
        return self.get_frame_rgb()

    def get_frame_rgb(self, ):
        return self.env.physics.render(camera_id=0, height=200, width=200)
