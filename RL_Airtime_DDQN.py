import csv
from time import sleep, time

import numpy as np
from DDQN_Agent_alex import Agent
from InfluxDBController import InfluxDBController
from utils import plotLearning

class Statistics(object):
    def __init__(self, slices):
        self.precision = 6

        self.log_file = open("Throughput_E5.csv", "w")
        self.log_data = csv.writer(self.log_file, delimiter=',')
        csv_header = ['Slice_{}_Throughput'.format(n) for n in slices] + \
                        ['Slice_{}_Quantum'.format(n) for n in slices] + \
                        ['Slice_{}_Action'.format(n) for n in slices]  + \
                        ['Reward', 'Mean_Over_Time_in_Seconds']
        self.log_data.writerow(csv_header)

        self.reset()


    def reset(self):
        self.throughputs = []
        self.quantums = []
        self.means = []
        self.rewards = []
        self.actions = []

    
    def storeTimestep(self, throughputs, quantums, reward, action):
        self.throughputs.append([round(tp, self.precision) for tp in throughputs])
        self.quantums.append([round(q, self.precision) for q in quantums])
        self.rewards.append(reward)
        self.actions.append(action)

        means = [round(np.mean(slice_throughputs), self.precision) for slice_throughputs in np.transpose(self.throughputs)]
        self.means.append(means)

        print(self.throughputs[-1], self.quantums[-1], self.actions[-1], self.rewards[-1])
        data = self.throughputs[-1] + self.quantums[-1] + self.actions[-1].tolist() + [self.rewards[-1]] 
        self.log_data.writerow(data)


    def generatePlots(self, slices):
        x = [k + 1 for k in range(len(self.means))]
        for i in range(slices):
            plotLearning(x, self.means[-1][i], self.quantums[-1][i], filename='TH'+str(i+1)+' and Q'+str(i+1)+' vs Time.png')


class Controller(object):
    def __init__(self, learning_rate, slices, required_throughputs):
        self.agent = Agent(alpha=learning_rate, gamma=0.5, slices=slices, epsilon=0.9, batch_size=64, input_dims=len(slices))
        self.influxController = InfluxDBController()
        self.slices = slices

        #pdb.set_trace()
        #agent.load_model()
        
        self.quantums = [1 / len(slices) for _ in slices]
        self.threshold_requirements = required_throughputs
        threshold_requirements_sum = sum(self.threshold_requirements)
        self.bandwidth_fractions = [req / threshold_requirements_sum for req in self.threshold_requirements]
        
        self.agent.update_quantums(self.quantums, self.slices)

    
    def run(self, n_episodes):
        statistics = Statistics(self.slices)
        interval = 30

        for i in range(n_episodes):
            throughputs = self.influxController.get_stats()
            throughputs = np.asarray(throughputs)
            excess_throughputs = throughputs - self.threshold_requirements
            all_throughputs_met = all([throughput > requirement for (throughput, requirement) in zip(throughputs, self.threshold_requirements)])

            action = self.agent.choose_action(excess_throughputs)
            if not all_throughputs_met or i < 2000:
                new_quantums = self.agent.execute_action(action, self.quantums, i, self.bandwidth_fractions)
                self.quantums = new_quantums
            
            sleep(interval - time() % interval)

            new_throughputs = self.influxController.get_stats()
            new_throughputs = np.asarray(throughputs)
            new_excess_throughputs = new_throughputs - self.threshold_requirements
            
            if not all_throughputs_met or i < 2000:
                reward = self.agent.get_reward(new_excess_throughputs)
                self.agent.remember(throughputs, action, reward, new_excess_throughputs, int(False))
                #pdb.set_trace()
                loss = self.agent.learn()

            #   	 pdb.set_trace()

            statistics.storeTimestep(new_excess_throughputs, self.quantums, reward, self.agent.action_possibilities[action])
            statistics.writeEpisodeStats()

            if i % 100 == 0 and i > 0:
                self.agent.save_model()
        #    if i % 500 == 0 and i > 0:
                
                #dd.writerow(Ts)
                #   	   print (eps_history)
                #   	    print('episode: ', i,'Throughput1: %.2f' % Throughput1,
                #   	          ' average score %.2f' % avg_score)
                #   	     if i % 10 == 0 and i > 0:
                #   	         agent.save_model()

        statistics.generatePlots()     
        # startlearning()
        # def launch(context, service_id, every=EVERY):
        #    """ Initialize the module. """

        #    return InfluxDBStats(context=context, service_id=service_id,
        #                       every=every)



if __name__ == '__main__':
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
    controller = Controller(learning_rate=0.005, slices=slice_ids, required_throughputs=slice_required_throughputs)
    controller.run(n_episodes=10000)