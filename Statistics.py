import csv
import numpy as np
from utils import plotLearning

class Statistics(object):
    def __init__(self, slices, filename):
        self.precision = 6

        self.log_file = open(filename, "w")
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

        data = self.throughputs[-1] + self.quantums[-1] + self.actions[-1].tolist() + [self.rewards[-1]] 
        self.log_data.writerow(data)


    def generatePlots(self, slices):
        x = [k + 1 for k in range(len(self.means))]
        for i in range(slices):
            plotLearning(x, self.means[-1][i], self.quantums[-1][i], filename='TH'+str(i+1)+' and Q'+str(i+1)+' vs Time.png')