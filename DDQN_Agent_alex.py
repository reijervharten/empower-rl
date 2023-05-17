import json
import pdb
import numpy as np
import requests
from influxdb import InfluxDBClient
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


class InfluxDBStats(object):
    """Instantiate a connection to the InfluxDB."""

    def __init__(self):
        self.host = 'localhost'
        self.port = 8086
        self.user = 'root'
        self.password = 'root'
        self.dbname = 'empower'
        self.pikey2 = 'B8:27:EB:95:92:76' #Pi_Key2 MAC 'E4:5F:01:12:AB:D1'
        self.pikey1 = 'B8:27:EB:43:C0:54'  # Ubuntu Laptop 'A4:17:31:72:41:11' Pikey1 >>E4:5F:01:12:AB:38
        self.pi4 = 'DC:A6:32:FC:81:A9'  # Adaptor 'E8:4E:06:23:9B:EC My Mobile '98:F6:21:F6:6E:74
        self.query1 = 'SELECT mean("tx_bps") FROM "empower.apps.lvapbincounter.lvapbincounter" WHERE ("sta"=$var_name) AND time >= now() - 30s and time <= now()'
        self.query2 = 'SELECT mean("tx_bps") FROM "empower.apps.lvapbincounter.lvapbincounter" WHERE ("sta"=$var_name) AND time >= now() - 30s and time <= now()'
        self.query3 = 'SELECT mean("tx_bps") FROM "empower.apps.lvapbincounter.lvapbincounter" WHERE ("sta"=$var_name) AND time >= now() - 30s and time <= now()'
        self.client = InfluxDBClient(self.host, self.port, self.user, self.password, self.dbname)

    #       print ("Connected to Influx DB Empower")

    def get_stats(Q1, Q2, Q3, C, pikey1, pi4, pikey2):
        #        self.Q1=stats.query1
        #        self.Q2=stats.query2
        #        self.Q3=stats.query3
        #        self.C=stats.client
        T1 = C.query(Q1, bind_params={"var_name": pikey1})
        T2 = C.query(Q2, bind_params={"var_name": pi4})
        T3 = C.query(Q3, bind_params={"var_name": pikey2})
        y1 = T1.raw
        y1 = y1['series']
        y1 = y1[0]
        y1 = y1['values']
        y1 = y1[0]
        y1 = y1[1]
        y2 = T2.raw
        y2 = y2['series']
        y2 = y2[0]
        y2 = y2['values']
        y2 = y2[0]
        y2 = y2[1]
        y3 = T3.raw
        y3 = y3['series']
        y3 = y3[0]
        y3 = y3['values']
        y3 = y3[0]
        y3 = y3[1]

        #        print("Querying data: " + Q1)
        TH1 = y1 * 8
        TH2 = y2 * 8
        TH3 = y3 * 8
        return TH1, TH2, TH3


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):#, fc3_dims, fc4_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        #Dense(fc3_dims),
        #Activation('relu'),
        #Dense(fc4_dims),
        #Activation('relu'),
        Dense(n_actions)])

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=100000, fname='ddqn_model.h5', replace_target=10):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)#, 128, 64)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)#, 128, 64)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            #            pdb.set_trace()
            action = np.argmax(actions)

        return action

    def measure_state(self, Th1, Th2, Th3):
        if Th1 > 0 and Th1 < 2: Th1_s = 1
        elif Th1 > 2 and Th1 < 4: Th1_s = 2
        elif Th1 > 4 and Th1 < 6: Th1_s = 3
        elif Th1 > 6 and Th1 < 8: Th1_s = 4
        elif Th1 > 8 and Th1 < 10: Th1_s = 5
        else: Th1_s = 6
    
        if Th2 > 0 and Th2 < 2: Th2_s = 1
        elif Th2 > 2 and Th2 < 4: Th2_s = 2
        elif Th2 > 4 and Th2 < 6: Th2_s = 3
        elif Th2 > 6 and Th2 < 8: Th2_s = 4
        elif Th2 > 8 and Th2 < 10: Th2_s = 5
        else: Th2_s = 6
    
        if Th3 > 0 and Th3 < 2: Th3_s = 1
        elif Th3 > 2 and Th3 < 4: Th3_s = 2
        elif Th3 > 4 and Th3 < 6: Th3_s = 3
        elif Th3 > 6 and Th3 < 8: Th3_s = 4
        elif Th3 > 8 and Th3 < 10: Th3_s = 5
        else: Th3_s = 6
    
        return Th1_s, Th2_s, Th3_s

    def execute_action(self, action, Q_i1, Q_i2, Q_i3, All_Throughput_met, TH1_met, TH2_met, TH3_met, iteration, MBB, BE, BG):
        action_possiblities = [("L", "L", "L"), ("L", "L", "H"), ("L", "H", "L"), ("L", "H", "H"), ("H", "L", "L"),
                               ("H", "L", "H"), ("H", "H", "L"), ("H", "H", "H"), ("N", "L", "L"), ("N", "L", "H"),
                               ("N", "H", "L"), ("N", "H", "H"), ("L", "N", "L"), ("L", "N", "H"), ("H", "N", "L"),
                               ("H", "N", "H"), ("L", "L", "N"), ("L", "H", "N"), ("H", "L", "N"), ("H", "H", "N"),
                               ("N", "N", "L"), ("N", "N", "H"), ("N", "L", "N"), ("N", "H", "N"), ("L", "N", "N"),
                               ("H", "N", "N"), ("N", "N", "N")]
        action_taken = action_possiblities[action]
        project_id = '304349b2-b610-4d10-bc82-b055613a8cf1'  # Project ID of empower SSID of Controller in Office
        default_slice=3
        slice_id1 = 0
        slice_id2 = 1
        slice_id3 = 2
        Airtime = 20000  # Airtime of 10 ms time slot which will be shared among all slices
        Q_min = 0.1
        Q_max = 0.99
        Q_Inc = 0.1
        Q_dec = 0.1
        if iteration > 500:
            Q_Inc = 0.05
            Q_dec = 0.02    
        elif iteration > 3000:
            Q_Inc = 0.005
            Q_dec = 0.005
        if Q_i1 + Q_i2 + Q_i3 > 1.0:
            if Q_i1 > 0.2:
                Q_i1 = Q_i1 - 0.3*Q_i1
            if Q_i2 > 0.2:
                Q_i2 = Q_i2 - 0.3*Q_i2
            if Q_i3 > 0.2:
                Q_i3 = Q_i3 - 0.3*Q_i3

        if action_taken == ("L", "L", "L"):
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("L", "L", "H"):  # 1
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3 + BG*Q_Inc
        elif action_taken == ("L", "H", "L"):  # 2
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2 + BE * Q_Inc
            Q_i3 = Q_i3 - Q_dec
        elif action_taken == ("L", "H", "H"):  # 3
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2 + BE*Q_Inc
            Q_i3 = Q_i3 + BG*Q_Inc
        elif action_taken == ("H", "L", "L"):  # 4
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("H", "L", "H"):  # 5
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3 + BG * Q_Inc
        elif action_taken == ("H", "H", "L"):  # 6
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2 + BE * Q_Inc
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("H", "H", "H"):  # 7
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2 + BE * Q_Inc
            Q_i3 = Q_i3 + BG * Q_Inc
        elif action_taken == ("N", "L", "L"):  # 8
            Q_i1 = Q_i1
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("N", "L", "H"):  # 9
            Q_i1 = Q_i1
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3 + BG * Q_Inc
        elif action_taken == ("N", "H", "L"):  # 10
            Q_i1 = Q_i1
            Q_i2 = Q_i2 + BE * Q_Inc
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("N", "H", "H"):  # 11
            Q_i1 = Q_i1
            Q_i2 = Q_i2 + BE*Q_Inc
            Q_i3 = Q_i3 + BG*Q_Inc
        elif action_taken == ("L", "N", "L"):  # 12
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("L", "N", "H"):  # 13
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2
            Q_i3 = Q_i3 + BG * Q_Inc
        elif action_taken == ("H", "N", "L"):  # 14
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("H", "N", "H"):  # 15
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2
            Q_i3 = Q_i3 + BG * Q_Inc
        elif action_taken == ("L", "L", "N"):  # 16
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3
        elif action_taken == ("L", "H", "N"):  # 17
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2 + BE*Q_Inc
            Q_i3 = Q_i3
        elif action_taken == ("H", "L", "N"):  # 18
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3
        elif action_taken == ("H", "H", "N"):  # 19
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2 + BE * Q_Inc
            Q_i3 = Q_i3
        elif action_taken == ("N", "N", "L"):  # 20
            Q_i1 = Q_i1
            Q_i2 = Q_i2
            Q_i3 = Q_i3 - BG*Q_dec
        elif action_taken == ("N", "N", "H"):  # 21
            Q_i1 = Q_i1
            Q_i2 = Q_i2
            Q_i3 = Q_i3 + BG * Q_Inc
        elif action_taken == ("N", "L", "N"):  # 22
            Q_i1 = Q_i1
            Q_i2 = Q_i2 - BE*Q_dec
            Q_i3 = Q_i3
        elif action_taken == ("N", "H", "N"):  # 23
            Q_i1 = Q_i1
            Q_i2 = Q_i2 + BE * Q_Inc
            Q_i3 = Q_i3
        elif action_taken == ("L", "N", "N"):  # 24
            Q_i1 = Q_i1 - MBB*Q_dec
            Q_i2 = Q_i2
            Q_i3 = Q_i3
        elif action_taken == ("H", "N", "N"):  # 25
            Q_i1 = Q_i1 + MBB * Q_Inc
            Q_i2 = Q_i2
            Q_i3 = Q_i3
        elif action_taken == ("N", "N", "N"):  # 26
            Q_i1 = Q_i1
            Q_i2 = Q_i2
            Q_i3 = Q_i3

        Q_i1 = Q_min if Q_i1 < Q_min else Q_i1
        Q_i2 = Q_min if Q_i2 < Q_min else Q_i2
        Q_i3 = Q_min if Q_i3 < Q_min else Q_i3
        Q_i1 = Q_max if Q_i1 > Q_max else Q_i1
        Q_i2 = Q_max if Q_i2 > Q_max else Q_i2
        Q_i3 = Q_max if Q_i3 > Q_max else Q_i3

        Q_i11 = Q_i1 * Airtime
        Q_i22 = Q_i2 * Airtime
        Q_i33 = Q_i3 * Airtime
        Q_d = Airtime-(Q_i11+Q_i22+Q_i33)
        if Q_d < 0:
            Q_d=0
        
        requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, default_slice),
                json={"properties": {"quantum": Q_d, "sta_scheduler": 2}}) # Station Scheduler is 1 for Deficit Round Robin and 2 for Airtime Deficit Round Robin

        requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id1),
                json={"properties": {"quantum": Q_i11, "sta_scheduler": 2}})

        requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id2),
                json={"properties": {"quantum": Q_i22, "sta_scheduler": 2}})

        requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id3),
                json={"properties": {"quantum": Q_i33, "sta_scheduler": 2}})
        return Q_i1, Q_i2, Q_i3

    def get_reward(self, Th1, Th2, Th3, TH1, TH2, TH3):
        self.Th1 = Th1
        self.Th2 = Th2
        self.Th3 = Th3
        self.TH1 = TH1
        self.TH2 = TH2
        self.TH3 = TH3
        if TH1 > Th1:
            a = 1
        else:
            a = 0
        if TH2 > Th2:
            b = 1
        else:
            b = 0
        if TH3 > Th3:
            c = 1
        else:
            c = 0
        rwd=(a, b, c)    
        if rwd == (1, 1, 1):
            r = 500
        else:
            r=-500
            
#        elif rwd == (1, 1, 0):
#            r = 100
#        elif rwd == (1, 0, 1):
#            r = 90
#        elif rwd == (0, 1, 1):
#            r = 80
#        elif rwd == (1, 0, 0):
#            r = 20
#        elif rwd == (0, 1, 0):
#            r = 10
#        elif rwd == (0, 0, 1):
#            r = 0                    
#        elif rwd == (0, 0, 0):
#            r = -500
        return r

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred
            #pdb.set_trace()
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            loss = self.q_eval.fit(state, q_target, verbose=1)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()
                
            return loss    

    def update_network_parameters(self):
        #pdb.set_trace()
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        
        if self.epsilon <= self.epsilon_min:
            self.update_network_parameters()
        
    def get_quantums(self, project_id, slice_id1, slice_id2, slice_id3):
        q1=requests.get('http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id1))
        q1=json.loads(q1.text)
        Q_S1=q1["properties"]["quantum"] 
        
        q2=requests.get('http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id2))
        q2=json.loads(q2.text)
        Q_S2=q2["properties"]["quantum"] 
        
        q3=requests.get('http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id3))
        q3=json.loads(q3.text)
        Q_S3=q3["properties"]["quantum"]
        
        return Q_S1, Q_S2, Q_S3
        
    def initialize_quantums(self, project_id, slice_id1, slice_id2, slice_id3, Initial_Quantum_Value):
        requests.put(
                    'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id1),
                    json={"properties": {"quantum": Initial_Quantum_Value, "sta_scheduler": 2}})
                    
        requests.put(
                    'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id2),
                    json={"properties": {"quantum": Initial_Quantum_Value, "sta_scheduler": 2}})
                    
        requests.put(
                    'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id3),
                    json={"properties": {"quantum": Initial_Quantum_Value, "sta_scheduler": 2}})    
