import os
import zmq
import json
import random
from collections import deque

import gym
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

import tensorflow as tf
import horovod.tensorflow.keras as hvd
import Experience_pb2 as exp
from tensorflow.python.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.95  # Discount Rate
        self.epsilon = 1.0  # Exploration Rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    # policy network
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        opt = Adam(lr=self.learning_rate * hvd.size())
        opt = hvd.DistributedOptimizer(opt)
        model.compile(loss='mse', optimizer=opt)
        return model

    # Add experience to buffer pool
    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # Policy network samples action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # Fetch a minibatch from buffer pool, apdate the weights
    def replay(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.75
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    agent = DQNAgent(state_size, action_size)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.17.0.13:2000")
    # socket.connect("tcp://localhost:2000")

    done = False
    batch_size = 32
    num_episodes = 1000
    weight = b''

    for e in range(num_episodes):
        print('Train episode {}'.format(e))
        for time in range(500):

            socket.send(weight)
            if len(weight):
                print('Model sent')
                weight = b''
            expt = exp.Experience()
            expt.ParseFromString(socket.recv())
            # print(expt.state, expt.action, expt.reward, expt.next_state, expt.done)
            agent.memorize(np.expand_dims(np.array(expt.state), axis=0), expt.action, expt.reward, np.expand_dims(np.array(expt.next_state), axis=0), expt.done)
            if expt.done:
                break
            if len(agent.replay_buffer) > batch_size:
                agent.replay(batch_size)

        if e % 5 == 0 and hvd.rank() == 0:
            agent.save('./save/cartpole-dqn_{}.h5'.format(e))
            # print("Model: cartpole-dqn_{}.h5 saved.".format(e))
            with open('./save/cartpole-dqn_{}.h5'.format(e), 'rb') as file:
                weight = file.read()
