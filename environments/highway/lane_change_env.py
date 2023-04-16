import numpy as np
from gym.envs.registration import register

from highway_env.envs import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle_V1

DEFAULT_SPEED = 5
HISTORY_LEN = 5
N_LANE = 2


class Lane_Change_Env(HighwayEnv):
    def __init__(self, config=None):
        self.other_vehicles = list()
        self.target_lane = None
        self.target_position_y = None
        super(Lane_Change_Env, self).__init__(config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "RgbObservation"
            },
            "action": {
                "type": "MetaAction",
            },
            "lanes_count": N_LANE,
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "duration": 200,
            'centering_position': [0.5, 0.5],
            "ego_spacing": 1,
            "vehicles_density": 1,
            "offroad_terminal": True,
            "policy_frequency": 4,
            "simulation_frequency": 15,
        })
        return config

    def _create_other_vehicles(self, controlled_vehicle):
        self.other_vehicles = list()
        n_vehicles = self.config["vehicles_count"]
        other_vehicles_type = Vehicle

        v_x0 = controlled_vehicle.position[0]
        for i in range(n_vehicles):
            new_vehicle = other_vehicles_type(self.road, self.target_lane.position(v_x0, 0),
                                              self.target_lane.heading_at(v_x0), speed=DEFAULT_SPEED)
            self.other_vehicles.append(new_vehicle)
            v_x0 = v_x0 + 40 + 8 * i
        return self.other_vehicles

    def _create_vehicles(self):
        # init target lanes
        _from = self.road.np_random.choice(list(self.road.network.graph.keys()))
        _to = self.road.np_random.choice(list(self.road.network.graph[_from].keys()))
        _lane_id = N_LANE - 1
        self.target_lane = self.road.network.get_lane((_from, _to, _lane_id))

        self.controlled_vehicles = []
        vehicle = Vehicle.create_random(
            self.road,
            speed=DEFAULT_SPEED,
            lane_id=0,
        )

        # init controlled vehicle
        vehicle_cls = self.action_type.vehicle_class
        if vehicle_cls == Vehicle:
            vehicle = vehicle_cls(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        else:
            vehicle = vehicle_cls(self.road, vehicle.position, vehicle.heading, vehicle.speed,
                                  target_speed=DEFAULT_SPEED, target_speeds=np.linspace(0, 20, 20))
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        # init other vehicles
        for i, vehicle in enumerate(self._create_other_vehicles(controlled_vehicle=vehicle)):
            # noinspection PyUnresolvedReferences
            if isinstance(vehicle, IDMVehicle):
                vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
        self.target_position_y = self.other_vehicles[0].position[1]

    def _check_terminal(self):
        history = self.vehicle.history
        if len(history) < HISTORY_LEN:
            return False
        for i in range(HISTORY_LEN):
            h = history[i]
            if h.lane_index[2] != N_LANE - 1 or np.abs(h.heading) > 0.1 or np.abs(self.target_position_y - h.position[1]) > 0.6:
                return False
        return True

    def _reward(self, action):
        """ The default reward
        """
        reward = 0
        if self.vehicle.speed < 5:
            reward -= 10
        if self.vehicle.speed > 15:
            reward -= 10
        if np.abs(self.vehicle.heading) > 0.8:
            reward -= 0
        if self.vehicle.lane_index[2] != N_LANE - 1:
            reward -= 1
        if self._check_terminal():
            reward += 1
        if self.vehicle.crashed or not self.vehicle.on_road:
            reward -= 1000
        return float(reward)

    def _is_terminal(self):
        return super(Lane_Change_Env, self)._is_terminal() or self._check_terminal()

    def _cost(self, action: int) -> float:
        return super(Lane_Change_Env, self)._cost(action)

    def _info(self, obs, action):
        info = super(Lane_Change_Env, self)._info(obs, action)
        info.update({'dist_to_target': np.abs(self.target_position_y - self.vehicle.position[1])})
        return info


register(
    id='lanechange-v0',
    entry_point='environments.highway.lane_change_env:Lane_Change_Env',
)
