import json
import pdb
import numpy as np
import requests
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam


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
    def __init__(self, alpha, gamma, slices, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=100000, fname='ddqn_model.h5', replace_target=10):
        self.slices = slices
        self.n_actions = 3 ** slices
        self.action_space = [i for i in range(self.n_actions)]
        self.action_possibilities = [[]]
        for i in range(self.slices):
            self.action_possibilities = [x + y for x in self.action_possibilities for y in [["L"], ["N"], ["H"]]]

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, self.n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, self.n_actions, input_dims, 256, 256)#, 128, 64)
        self.q_target = build_dqn(alpha, self.n_actions, input_dims, 256, 256)#, 128, 64)


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


    def execute_action(self, action, quantums, iteration, threshold_fractions):
        action_taken = self.action_possibilities[action]
        project_id = '2788d1eb-dbe4-4972-80be-e48462968265'  # Project ID of empower SSID of Controller in Office
        default_slice = 3
        slice_ids = [0, 1, 2]
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
 
        # if Q_i1 + Q_i2 + Q_i3 > 1.0:
        #     if Q_i1 > 0.2:
        #         Q_i1 = Q_i1 - 0.3*Q_i1
        #     if Q_i2 > 0.2:
        #         Q_i2 = Q_i2 - 0.3*Q_i2
        #     if Q_i3 > 0.2:
        #         Q_i3 = Q_i3 - 0.3*Q_i3

        for i in range(self.slices):
            match action_taken[i]:
                case "L":
                    quantums[i] -= threshold_fractions[i] * Q_dec
                case "N":
                    quantums[i] += 0
                case "H":
                    quantums[i] += threshold_fractions[i] * Q_Inc
            
            quantums[i] = Q_min if quantums[i] < Q_min else quantums[i]
            quantums[i] = Q_max if quantums[i] > Q_max else quantums[i]

        for i in range(self.slices):
            airtime = Airtime * quantums[i]
            slice_id = slice_ids[i]
            requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id),
                json={"properties": {"quantum": airtime, "sta_scheduler": 2}}) # Station Scheduler is 1 for Deficit Round Robin and 2 for Airtime Deficit Round Robin

        remaining_quantum = 1-sum(quantums)
        remaining_quantum = 0 if remaining_quantum < 0 else remaining_quantum
        requests.put(
                'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, default_slice),
                json={"properties": {"quantum": remaining_quantum, "sta_scheduler": 2}}) 
        
        return quantums


    def get_reward(self, threshold_requirements, throughputs):
        if all(throughputs[i] >= threshold_requirements[i] for i in range(self.slices)):
            return 500
        else:
            return -500


    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            # Use the Q target network to evaluate the Q value of the new state
            q_next = self.q_target.predict(new_state)

            # Use the Q evaluation network to guess the best action in the new state
            q_eval = self.q_eval.predict(new_state)
            max_actions = np.argmax(q_eval, axis=1)
            
            # Use the Q evaluation network as a base of what the target Q values should be
            q_pred = self.q_eval.predict(state)
            q_target = q_pred


            #pdb.set_trace()
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # Update the Q target values with the adjusted values found by taking the actions in the batch
            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done


            # Fit the Q evaluation network
            loss = self.q_eval.fit(state, q_target, verbose=1)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min

            # Replace the Q target network with the Q evaluation network every (self.replace_target) iterations
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

        
    # def get_quantums(self, project_id, slice_id1, slice_id2, slice_id3):
    #     q1=requests.get('http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id1))
    #     q1=json.loads(q1.text)
    #     Q_S1=q1["properties"]["quantum"] 
        
    #     q2=requests.get('http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id2))
    #     q2=json.loads(q2.text)
    #     Q_S2=q2["properties"]["quantum"] 
        
    #     q3=requests.get('http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id3))
    #     q3=json.loads(q3.text)
    #     Q_S3=q3["properties"]["quantum"]
        
    #     return Q_S1, Q_S2, Q_S3

        
    # def initialize_quantums(self, project_id, slice_id1, slice_id2, slice_id3, Initial_Quantum_Value):
    #     requests.put(
    #                 'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id1),
    #                 json={"properties": {"quantum": Initial_Quantum_Value, "sta_scheduler": 2}})
                    
    #     requests.put(
    #                 'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id2),
    #                 json={"properties": {"quantum": Initial_Quantum_Value, "sta_scheduler": 2}})
                    
    #     requests.put(
    #                 'http://foo:foo@localhost:8888/api/v1/projects/%s/wifi_slices/%s' %(project_id, slice_id3),
    #                 json={"properties": {"quantum": Initial_Quantum_Value, "sta_scheduler": 2}})    
