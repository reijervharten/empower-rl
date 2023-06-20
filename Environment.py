import random
from time import sleep
import time
import numpy as np
import requests

from InfluxDBController import InfluxDBController
from Statistics import Statistics

slice_ids = [0, 8, 18, 20, 30, 44, 46, 48]
slice_required_throughputs = [
    2, # DSCP 0: Best effort
    0.5, # DSCP 8: Low priority (Video surveillance)
    4, # DSCP 18: ??
    1.5, # DSCP 20: Network operations
    8, # DSCP 30: Video
    1, # DSCP 44: Voice
    1.5, # DSCP 46: Critical data
    0.5, # DSCP 48: Network control
]
episode_interval_seconds = 4

class Environment:
    def __init__(self):
        self.upper_bound = 10000
        self.lower_bound = 500

        self.num_slices = len(slice_ids)
        self.influxController = InfluxDBController()
        self.prev_test_no = "E23c"
        self.test_no = "E23d"
        self.statistics = Statistics(slice_ids, "Throughput_{}.csv".format(self.test_no))
        self.reset()

        self.prev_timestamp = time.time()

    def reset(self):
        self.quantums = [2500 for _ in slice_ids]
        # self.update_quantums()
        # self.prev_state = self.get_throughputs()
        self.prev_state = [4 for _ in slice_ids]

        return self.prev_state
    
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
            if not matching_pair == None:
                throughputs.append(matching_pair[1])
            else:
                print("Warning: no throughput found for slice {}".format(id))
                throughputs.append(None)

        return throughputs
    
    def calculate_reward(self, throughputs):
        reward = 100
        for id, tp, goal in zip(slice_ids, throughputs, slice_required_throughputs):
            if (tp < goal):
                tp = max(tp, 0)
                reward -= (goal - tp) * (goal - tp)
        
        return reward
    
    def step(self, action, simulate=False):
        self.quantums = action

        if simulate:
            throughput_sum = sum(self.prev_state) + (random.random() * 2 - 1) * 0.5 # Random value between -0.5 and 0.5
            throughput_sum = max(throughput_sum, 18)
            throughput_sum = min(throughput_sum, 22)
            quantum_sum = sum(action)
            new_throughputs = [throughput_sum * quantum / quantum_sum for quantum in action]
            reward = self.calculate_reward(new_throughputs)
        
        else:
            self.update_quantums()

            sum_throughputs = 0
            while sum_throughputs < 12:
                next_timestamp = self.prev_timestamp + episode_interval_seconds
                sleep_duration = next_timestamp - time.time()
                if (sleep_duration < 0):
                    print("Warning: overshot next timestamp by {} seconds".format(sleep_duration))
                    self.prev_timestamp = time.time()
                else:
                    sleep(sleep_duration)
                    self.prev_timestamp = next_timestamp

                new_throughputs = self.get_throughputs()
                if new_throughputs.count(None) > 0:
                    continue
                sum_throughputs = sum(new_throughputs)

            reward = self.calculate_reward(new_throughputs)        

        self.statistics.storeTimestep(new_throughputs, self.quantums, reward, action)

        state = new_throughputs
        self.prev_state = state
        return state, reward
    
    def approximate_next_state(self, state, action):
        total_throughput = sum(state)
        next_state = []

        #action += np.random.rand(len(action)) *200
        
        for action_value in action:
            fraction = action_value / sum(action)
            next_state.append(total_throughput * fraction)
        
        return next_state
    
    def approximate_reward(self, next_state):
        return self.calculate_reward(next_state)
    
