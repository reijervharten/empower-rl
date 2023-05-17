# from empower.apps.RL_Airtime.dqn_agent import Agent
import csv
# from tornado import gen
# from concurrent.futures import ThreadPoolExecutor
# import threading
from time import sleep, time

import numpy as np
import pdb
from DDQN_Agent_alex import Agent
# from empower_core.app import EApp
# from empower_core.app import EVERY
# from empower.apps.RL_Airtime.dqn_agent import InfluxDBStats
from DDQN_Agent_alex import InfluxDBStats
# from empower.apps.RL_Airtime.utils import plotLearning
from utils import plotLearning

# if __name__ == '__main__':
lr = 0.005
n_games = 7000
agent = Agent(alpha=lr, gamma=0.5, n_actions=27, epsilon=0.9, batch_size=64, input_dims=3)
#pdb.set_trace()
#agent.load_model()
Initial_Quantum_value=500
Q_i1 = 0.1
Q_i2 = 0.1
Q_i3 = 0.1
Threshold_TH1 = 4.0  # Threshold in Mbps
Threshold_TH2 = 1.5  # Threshold in Mbps
Threshold_TH3 = 3.0  # Threshold in Mbps
Agg_req=Threshold_TH1 + Threshold_TH2 + Threshold_TH3
MBB=Threshold_TH1 / Agg_req
BE=Threshold_TH2 / Agg_req
BG=Threshold_TH3 / Agg_req
Throughput1 = []  # For storing throughput values in each step
Throughput2 = []
Throughput3 = []
TH1_mean = []
TH2_mean = []
TH3_mean = []
Ts = []
quantums = []
eps_history = []
action_history = []
Q1_history = []
Q2_history = []
Q3_history = []
quantums =[0.1, 0.1, 0.1]
reward = [1]
i = 0
f = open("Throughput_E1.csv", "w")
log_data = csv.writer(f, delimiter=',')
csv_header=['Slice_1_Throughput','Slice_1_Quantum','Slice_2_Throughput','Slice_2_Quantum','Slice_3_Throughput','Slice_3_Quantum','Reward','Action', 'Mean_Over_Time_in_Seconds']
log_data.writerow(csv_header)
precision=1
# class DRL_Airtime_optimizer(EApp):

#    def __init__(self, context, service_id, every=EVERY):  

#        super().__init__(context=context,
#                         service_id=service_id,
#                         every=every)
stats = InfluxDBStats()
state_spaces = InfluxDBStats.get_stats(stats.query1, stats.query2, stats.query3, stats.client, stats.pikey1,
                                       stats.pi4, stats.pikey2)
TH1 = round(state_spaces[0]/1000000,precision)
TH2 = round(state_spaces[1]/1000000,precision)
TH3 = round(state_spaces[2]/1000000,precision)
state = [TH1, TH2, TH3]
filename1 = 'TH1 and Q1 vs Time.png'
filename2 = 'TH2 and Q2 vs Time.png'
filename3 = 'TH3 and Q3 vs Time.png'
#    def loop (self):
interval = 1
j=0
for i in range(n_games):
    Run_every_interval_seconds = True
    All_Throughput_met = False
    TH1_met = False
    TH2_met = False
    TH3_met = False
    if TH1 > Threshold_TH1 and TH2 > Threshold_TH2 and TH3 > Threshold_TH3: 
        All_Throughput_met = True
    if TH1 > Threshold_TH1:
        TH1_met = True
    if TH2 > Threshold_TH2:
        TH2_met = True
    if TH3 > Threshold_TH3:
        TH3_met = True
    while Run_every_interval_seconds:
        done = True
        #Learning only when one of the slice throughputs is not met.  
        stats = InfluxDBStats()
        state_spaces = InfluxDBStats.get_stats(stats.query1, stats.query2, stats.query3, stats.client, stats.pikey1,
                                               stats.pi4, stats.pikey2)
        TH1_t = round(state_spaces[0]/1000000,precision)
        TH2_t = round(state_spaces[1]/1000000,precision)
        TH3_t = round(state_spaces[2]/1000000,precision)
        state = [TH1_t, TH2_t, TH3_t]
        state = np.array(state)
        action = agent.choose_action(state)
        if not All_Throughput_met or i < 2000:
            quantums = agent.execute_action(action, Q_i1, Q_i2, Q_i3, All_Throughput_met, TH1_met, TH2_met, TH3_met, i, MBB, BE, BG)
            Q_i1 = quantums[0]
            Q_i2 = quantums[1]
            Q_i3 = quantums[2]
        sleep(interval - time() % interval)
        state_spaces = InfluxDBStats.get_stats(stats.query1, stats.query2, stats.query3, stats.client, stats.pikey1,
                                           stats.pi4, stats.pikey2)
        TH1 = round(state_spaces[0]/1000000,precision)
        TH2 = round(state_spaces[1]/1000000,precision)
        TH3 = round(state_spaces[2]/1000000,precision)
        new_state = [TH1, TH2, TH3]
        if not All_Throughput_met or i < 2000:
            reward = agent.get_reward(Threshold_TH1, Threshold_TH2, Threshold_TH3, TH1, TH2, TH3)
            agent.remember(state, action, reward, new_state, int(done))
            state = new_state
            #pdb.set_trace()
            loss = agent.learn()
        #   	 pdb.set_trace()
        Q1_history.append(Q_i1)
        Q2_history.append(Q_i2)
        Q3_history.append(Q_i3)
        j=j+1
        Run_every_interval_seconds = False #Controlling the loop to run every Intervsa seconds
        #pdb.set_trace()
        Throughput1.append(TH1)
        Throughput2.append(TH2)
        Throughput3.append(TH3)
        #    	reward_history.append(reward)
#        action_history.append(action)
        TH1_m = np.mean(Throughput1[max(0, i - 1):(i + 1)])
        TH2_m = np.mean(Throughput2[max(0, i - 1):(i + 1)])
        TH3_m = np.mean(Throughput3[max(0, i - 1):(i + 1)])
        mean_over_time_in_seconds=15*interval
        TH1_mean.append(TH1_m)
        TH2_mean.append(TH2_m)
        TH3_mean.append(TH3_m)
    
    Q1=int(quantums[0]*20000)
    Q2=int(quantums[1]*20000)
    Q3=int(quantums[2]*20000)
    print(Q1, Q2, Q3)
    print("Mean Throughput of Slice 1 in Mbps:", TH1_m)
    print("Mean Throughput of Slice 2 in Mbps:", TH2_m)
    print("Mean Throughput of Slice 3 in Mbps:", TH3_m)
    print(i)
    data=(TH1_m, Q1, TH2_m, Q2, TH3_m, Q3, reward, action, mean_over_time_in_seconds)
    log_data.writerow(data)
    if j % 100 == 0 and j > 0:
        agent.save_model()
#    if i % 500 == 0 and i > 0:
        
        #dd.writerow(Ts)
        #   	   print (eps_history)
        #   	    print('episode: ', i,'Throughput1: %.2f' % Throughput1,
        #   	          ' average score %.2f' % avg_score)
        #   	     if i % 10 == 0 and i > 0:
        #   	         agent.save_model()
    if i % 150 == 0 and i >0:
        x = [k + 1 for k in range(i+1)]
        plotLearning(x, TH1_mean, Q1_history, filename1)
        plotLearning(x, TH2_mean, Q2_history, filename2)
        plotLearning(x, TH3_mean, Q3_history, filename3)
        
f.close()        
# startlearning()
# def launch(context, service_id, every=EVERY):
#    """ Initialize the module. """

#    return InfluxDBStats(context=context, service_id=service_id,
#                       every=every)
