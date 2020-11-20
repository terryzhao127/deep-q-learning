import os
import zmq
import json
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

import exp_pb2 as exp
from Wrappers import *
from utils.protobuf_tool import bytes2arr
import horovod.tensorflow.keras as hvd
from tensorflow.python.keras import backend as K

# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.99  # Discount Rate
        self.epsilon = 1.0  # Exploration Rate
        self.epsilon_min = 0.02
        self.epsilon_decay = .999985
        self.learning_rate = 1e-4
        self.model = self._build_model()

    # policy network
    def _build_model(self):
        # input: (B, 4, 84, 84)
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
        # (B, 84, 84, 4)
        model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
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
    env = gym.make("PongNoFrameskip-v4")
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)

    state_size = env.observation_space.shape # (4, 84, 84)
    action_size = env.action_space.n    # 6

    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    agent = DQNAgent(state_size, action_size)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.17.0.10:2000")
    # socket.connect("tcp://localhost:2000")

    done = False
    batch_size = 32
    num_episodes = 10000
    weight = b''

    for e in range(num_episodes):
        print('Training episode {}'.format(e))
        while True:
            socket.send(weight)
            if len(weight):
                print('Model sent')
                weight = b''
            expt = exp.exp()
            expt.ParseFromString(socket.recv())
            # print(expt.state, expt.action, expt.reward, expt.next_state, expt.done)
            # print(expt.reward, expt.done)
            agent.memorize(bytes2arr(expt.state), expt.action, expt.reward, bytes2arr(expt.next_state), expt.done)
            if expt.done:
                break
            if len(agent.replay_buffer) > batch_size:
                agent.replay(batch_size)

        if e % 5 == 0 and hvd.rank() == 0:
        # if e % 1 == 0:
            agent.save('./save/pong-dqn_{}.h5'.format(e))
            print("Model: pong-dqn_{}.h5 saved.".format(e))
            with open('./save/pong-dqn_{}.h5'.format(e), 'rb') as file:
                weight = file.read()