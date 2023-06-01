from time import sleep
import requests

from InfluxDBController import InfluxDBController
from RL_Airtime_DDQN import Statistics

slice_ids = [0, 8, 16, 20, 30, 44, 46, 48]
slice_required_throughputs = [
    1, # DSCP 0: Best effort
    2, # DSCP 8: Low priority (Video surveillance)
    1, # DSCP 16: ??
    0.5, # DSCP 20: Network operations
    2.5, # DSCP 30: Video
    1, # DSCP 44: Voice
    1.5, # DSCP 46: Critical data
    0.5, # DSCP 48: Network control
]
episode_interval_seconds = 30

class Environment:
    def __init__(self):
        self.influxController = InfluxDBController()
        self.statistics = Statistics(slice_ids)
        self.reset()

    def reset(self):
        self.quantums = [12500 for _ in slice_ids]
        self.update_quantums()

        return self.quantums
    
    def update_quantums(self):
        project_id = '2788d1eb-dbe4-4972-80be-e48462968265'  # Project ID of empower SSID of Controller in Office

        for (slice_id, quantum) in zip(slice_ids, self.quantums):
            requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id),
                json={"properties": {"quantum": quantum, "sta_scheduler": 2}}) # Station Scheduler is 1 for Deficit Round Robin and 2 for Airtime Deficit Round Robin
            
    def step(self, action):
        self.quantums = action
        self.update_quantums()

        sleep(episode_interval_seconds)
        new_throughputs = self.influxController.get_stats()
        reward = 0
        for id, tp, goal in zip(slice_ids, new_throughputs, slice_required_throughputs):
            if tp > goal:
                if tp < 1.1 * goal:
                    reward += 1
            else:
                tp = max(tp, 0)
                weighted_error = id * (goal - tp) / (50 * goal)
                reward -= weighted_error * weighted_error

        self.statistics.storeTimestep(new_throughputs, self.quantums, reward, action)

        state = self.quantums
        return state, reward
    
