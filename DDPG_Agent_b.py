import csv
import os
import time
import keras
from keras import layers
import tensorflow as tf
import numpy as np

from Environment import Environment


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')
        
        self.fc1 = layers.Dense(self.fc1_dims, activation='relu')
        self.fc2 = layers.Dense(self.fc2_dims, activation='relu')
        self.q = layers.Dense(1, activation=None)
    
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        
        q = self.q(action_value)
        
        return q
    
class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, fc1=400, fc2=300, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        # self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.upper_bound
        self.min_action = env.lower_bound

        # self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        # self.target_critic = CriticNetwork(name='target_critic')

        self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=beta), loss='mean_squared_error')
        # self.target_critic.compile(optimizer=keras.optimizers.Adam(learning_rate=beta), loss='mean_squared_error')

        # self.target_critic.set_weights(self.critic.get_weights())
    
    def updateFromBatch(self, states, actions, rewards, new_states):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards
            critic_loss = keras.losses.MSE(target, critic_value)
        
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))


env = Environment()
num_states = env.num_slices
num_actions = env.num_slices
upper_bound = env.upper_bound
lower_bound = env.lower_bound

agent = Agent(num_states, env=env, n_actions=num_actions)

t0 = time.time()
log_file = open('model_critic.csv', "w")
log_data = csv.writer(log_file, delimiter=',')
csv_header = ['Test_{}'.format(n) for n in range(5)]
log_data.writerow(csv_header)

for i in range(100000):
    states = []
    actions = []

    for _ in range(64):
        throughput_sample = [0.3,0.3,0.3]
        offsets = np.random.rand(num_states) * 0.2 - 0.1 # Values between -0.1 and 0.1
        state = [tp + offset for tp, offset in zip(throughput_sample, offsets)]
        action = np.random.rand(num_actions) * upper_bound - lower_bound #policy(tf.expand_dims(tf.convert_to_tensor(state), 0), ou_noise)

        states.append(state)
        actions.append(action)

    next_states = [env.approximate_next_state(state, action) for state, action in zip(states, actions)]
    rewards = [env.approximate_reward(next_state) for next_state in next_states]

    agent.updateFromBatch(states, actions, rewards, next_states)

    if i % 1000 == 0:
        print('i : ' + str(i) + ' / 100000, time: ' + str(time.time() - t0) + ' seconds')
        agent.critic.save_weights('model_critic_i_' + str(i) + '.h5')


        states = [[0.3,0.3,0.3] for _ in range(5)]
        actions = [[0.0, 0.0, 0.0], [12500.0, 12500.0, 12500.0], [50000.0, 50000.0, 50000.0], [50000.0, 30000.0, 0.0], [0.0, 30000.0, 50000.0]]

        states = tf.convert_to_tensor(states,  dtype=tf.float32)
        actions = tf.convert_to_tensor(actions,  dtype=tf.float32)
        critic_value = tf.squeeze(agent.critic(states, actions), 1)
        
        print(critic_value.numpy().tolist())
        log_data.writerow(critic_value.numpy().tolist())