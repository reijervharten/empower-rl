from time import sleep
import requests

from InfluxDBController import InfluxDBController
from Statistics import Statistics

slice_ids = [0, 8, 16] #, 20, 30, 44, 46, 48]
slice_required_throughputs = [
    0, # DSCP 0: Best effort
    4, # DSCP 8: Low priority (Video surveillance)
    8]#, # DSCP 16: ??
#     0.5, # DSCP 20: Network operations
#     2.5, # DSCP 30: Video
#     1, # DSCP 44: Voice
#     1.5, # DSCP 46: Critical data
#     0.5, # DSCP 48: Network control
# ]
episode_interval_seconds = 10

class Environment:
    def __init__(self):
        self.num_slices = len(slice_ids)
        self.influxController = InfluxDBController()
        self.statistics = Statistics(slice_ids, "Throughput_E8.csv")
        self.reset()

    def reset(self):
        self.quantums = [12500 for _ in slice_ids]
        self.update_quantums()

        return self.get_throughputs()
    
    def update_quantums(self):
        project_id = '2788d1eb-dbe4-4972-80be-e48462968265'  # Project ID of empower SSID of Controller in Office

        for (slice_id, quantum) in zip(slice_ids, self.quantums):
            requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id),
                json={"properties": {"quantum": quantum, "sta_scheduler": 2}}) # Station Scheduler is 1 for Deficit Round Robin and 2 for Airtime Deficit Round Robin
    
    def get_throughputs(self):
        throughputs = []
        throughput_pairs = self.influxController.get_stats()
        for id in slice_ids:
            matching_pair = next(filter(lambda pair: pair[0] == str(id), throughput_pairs), None)
            throughputs.append(matching_pair[1])
        return throughputs
    
    def calculate_reward(self, throughputs):
        reward = 300
        for id, tp, goal in zip(slice_ids, throughputs, slice_required_throughputs):
            tp = max(tp, 0)
            weighted_error = (goal - tp)
            reward -= 100*weighted_error * weighted_error
        
        return reward
    
    def step(self, action):
        self.quantums = action
        self.update_quantums()

        sleep(episode_interval_seconds)
        new_throughputs = self.get_throughputs()

        reward = self.calculate_reward(new_throughputs)        

        self.statistics.storeTimestep(new_throughputs, self.quantums, reward, action)

        state = new_throughputs
        return state, reward
    
    def approximate_next_state(self, state, action):
        total_throughput = sum(state)
        next_state = []
        
        for action_value in action:
            fraction = action_value / sum(action)
            next_state.append(total_throughput * fraction)
        
        return next_state
    
    def approximate_reward(self, next_state):
        return self.calculate_reward(next_state)
    
