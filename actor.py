import os
import zmq
import random
from io import BytesIO
from collections import deque

import gym
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from data_pb2 import Data


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
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5000")

    if not os.path.exists('save_actor'):
        os.makedirs('save_actor')

    cnt = 0
    done = False
    batch_size = 32
    num_episodes = 1000
    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            weight = socket.recv()
            if len(weight):
                with open('save_actor/cartpole-{}.h5'.format(e), 'wb') as f:
                    f.write(weight)
                agent.load('save_actor/cartpole-{}.h5'.format(e))

            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            data = Data(state=arr2bytes(state), next_state=arr2bytes(next_state), action=int(action),
                        reward=reward, done=done, epoch=e)

            socket.send(data.SerializeToString())

            cnt += 1
            state = next_state
            if done:
                print('episode: {}/{}, score: {}, e: {:.2}'.format(e+1, num_episodes, time, agent.epsilon))
                break
            if cnt > batch_size:
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
