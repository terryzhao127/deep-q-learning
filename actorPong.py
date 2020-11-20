import zmq
import json
import random
import os
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

import exp_pb2 as exp
from Wrappers import *
from utils.protobuf_tool import arr2bytes
from tensorboardX import SummaryWriter

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
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
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

    # Note..
    def replay(self, batch_size):
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

    agent = DQNAgent(state_size, action_size)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:2000")

    if not os.path.exists('actorRecv'): os.mkdir('actorRecv')

    writer = SummaryWriter(comment="-" + "PongNoFrameskip-v4")

    cnt = 0
    done = False
    batch_size = 32
    num_episodes = 10000
    total_rewards = []
    re = 0.0
    for e in range(num_episodes):
        state = env.reset()
        state = np.expand_dims(state, 0)
        while True:
            weight = socket.recv()
            if len(weight):
                print('Model received')
                with open('actorRecv/pong-dqn_{}.h5'.format(e), 'wb') as file:
                    file.write(weight)
                agent.load('actorRecv/pong-dqn_{}.h5'.format(e))

            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            re += reward
            # print(reward, re, done)
            next_state = np.expand_dims(next_state, 0)

            expt = exp.exp()
            expt.state = arr2bytes(state)
            expt.action = int(action)
            expt.reward = reward
            expt.next_state = arr2bytes(next_state)
            expt.done = done
            cnt += 1
            # print(expt.state, expt.action, expt.reward, expt.next_state, expt.done)
            socket.send(expt.SerializeToString())

            state = next_state
            if done:
                total_rewards.append(re)
                mean_reward = np.mean(total_rewards[-100:])
                print('episode: {}/{}, mean reward {}, reward: {}, epsilon: {:.2}'.format(e, num_episodes, mean_reward, re, agent.epsilon))
                re = 0.0
                writer.add_scalar("reward_100", mean_reward, e)
                writer.add_scalar("reward", reward, e)
                break

            if cnt > batch_size:
                agent.replay(batch_size)

    writer.close()