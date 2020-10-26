import os
import zmq
import random
from io import BytesIO
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import horovod.tensorflow.keras as hvd

from data_pb2 import Data


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


def arr2bytes(arr):
    arr_bytes = BytesIO()
    np.save(arr_bytes, arr, allow_pickle=False)
    return arr_bytes.getvalue()


def bytes2arr(arr_bytes):
    arr = np.load(BytesIO(arr_bytes), allow_pickle=False)
    return arr


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

    def _build_model(self):
        """Build Neural Net for Deep Q-learning Model"""

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.compile(loss='mse', optimizer=hvd.DistributedOptimizer(Adam(lr=self.learning_rate)))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

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

    agent = DQNAgent(state_size, action_size)
    # agent.load('./save/cartpole-dqn.h5')

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.17.0.2:5000")

    if hvd.rank() == 0:
        if not os.path.exists('save_learner'):
            os.mkdir('save_learner')

    done = False
    batch_size = 32
    num_episodes = 1000
    weight = b''
    e = 0
    while e < num_episodes:
        print('Train episode {}'.format(e+1))
        for time in range(500):

            socket.send(weight)
            weight = b''

            data = Data()
            data.ParseFromString(socket.recv())
            state, next_state = bytes2arr(data.state), bytes2arr(data.next_state)
            agent.memorize(state, data.action, data.reward, next_state, data.done)
            if data.epoch > e:
                e = data.epoch
                break
            if len(agent.replay_buffer) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0 and hvd.rank() == 0:
            agent.save('save_learner/cartpole-{}.h5'.format(e))
            with open('save_learner/cartpole-{}.h5'.format(e), 'rb') as f:
                weight = f.read()
