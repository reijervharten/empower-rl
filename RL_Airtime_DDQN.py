import csv
from time import sleep, time

import numpy as np
import pdb
from DDQN_Agent_alex import Agent
from InfluxDBController import InfluxDBController
from utils import plotLearning

class Statistics(object):
    def __init__(self, slices):
        self.precision = 1

        self.slices = slices

        self.log_file = open("Throughput_E1.csv", "w")
        self.log_data = csv.writer(self.log_file, delimiter=',')
        csv_header = ['Slice_{}_Throughput'.format(n) for n in range(slices)] + \
                        ['Slice_{}_Quantum'.format(n) for n in range(slices)] + \
                        ['Reward', 'Action', 'Mean_Over_Time_in_Seconds']
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

        means = [round(np.mean(slice_throughputs), self.precision) for slice_throughputs in np.transpose(throughputs)]
        self.means.append(means)

    
    def writeEpisodeStats(self):
        print("quantums:", [x*20000 for x in self.quantums[-1]])
        print("mean throughputs:", self.means[-1])
        data = self.means[-1] + [x*20000 for x in self.quantums[-1]] + [self.rewards[-1]] + [self.actions[-1]]
        self.log_data.writerow(data)


    def generatePlots(self, slices):
        x = [k + 1 for k in range(len(self.means))]
        for i in range(slices):
            plotLearning(x, self.means[-1][i], self.quantums[-1][i], filename='TH'+str(i+1)+' and Q'+str(i+1)+' vs Time.png')


class Controller(object):
    def __init__(self, learning_rate, slices=3):
        self.agent = Agent(alpha=learning_rate, gamma=0.5, slices=slices, epsilon=0.9, batch_size=64, input_dims=3)
        self.influxController = InfluxDBController()
        self.slices = slices

        #pdb.set_trace()
        #agent.load_model()
        
        self.quantums = [0.1 for i in range(slices)]
        self.threshold_requirements = [4.0, 1.5, 3.0]  # Threshold in Mbps
        threshold_requirements_sum = sum(self.threshold_requirements)
        self.threshold_fractions = [req / threshold_requirements_sum for req in self.threshold_requirements]
        
        self.precision = 1

    
    def run(self, n_episodes):
        statistics = Statistics(self.slices)
        interval = 1
        iterations = 0

        for i in range(n_episodes):
            Run_every_interval_seconds = True

            state_spaces = self.influxController.get_stats()
            throughputs = [tp / 1000000 for tp in state_spaces]
            all_throughputs_met = all([throughput > requirement for (throughput, requirement) in zip(throughputs, self.threshold_requirements)])

            while Run_every_interval_seconds:
                done = True
                #Learning only when one of the slice throughputs is not met.  

                state_spaces = self.influxController.get_stats()
                throughputs = [tp / 1000000 for tp in state_spaces]
                throughputs = np.asarray(throughputs)

                action = self.agent.choose_action(throughputs)
                if not all_throughputs_met or i < 2000:
                    new_quantums = self.agent.execute_action(action, self.quantums, i, self.threshold_fractions)
                    self.quantums = new_quantums
                
                sleep(interval - time() % interval)

                state_spaces = self.influxController.get_stats()
                new_throughputs = [tp / 1000000 for tp in state_spaces]
                new_throughputs = np.asarray(throughputs)
                
                if not all_throughputs_met or i < 2000:
                    reward = self.agent.get_reward(self.threshold_requirements, new_throughputs)
                    self.agent.remember(throughputs, action, reward, new_throughputs, int(done))
                    throughputs = new_throughputs
                    #pdb.set_trace()
                    loss = self.agent.learn()

                #   	 pdb.set_trace()
                iterations += 1
                Run_every_interval_seconds = False #Controlling the loop to run every Intervsa seconds

                statistics.storeTimestep(throughputs, self.quantums, reward, self.agent.action_possibilities[action])
            
            statistics.writeEpisodeStats()

            if iterations % 100 == 0 and iterations > 0:
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
    controller = Controller(learning_rate=0.005, slices=3)
    controller.run(n_episodes=10000)