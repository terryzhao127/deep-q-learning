import random
from collections import deque

import gym
import zmq
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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
        #open('agent_data','a').write(str((state, action, reward, next_state, done)) + '\n')
        self.replay_buffer.append((state, action, reward, next_state, done))
        #message = np.array((state, action, reward, next_state, done), dtype = object)
        #socket.send(message)

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
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:6566")

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    # agent.load('./save/cartpole-dqn.h5')

    done = False
    batch_size = 32
    num_episodes = 1000
    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        send_flag = 0
        for time in range(500):
            message = socket.recv()
            if e % 2 == 0:
                if send_flag == 0:
                    agent.save('./save/cartpole-dqn{}.h5'.format(e))
                    with open('save/cartpole-dqn{}.h5'.format(e), 'rb') as f:
                        msend = f.read()
                    socket.send(msend)
                    send_flag = 1
                else:
                    socket.send(b"Cover")
            else:
                socket.send(b"Cover")
            message = eval(message)
            agent.memorize(np.array(message[0]), message[1], message[2], np.array(message[3]), message[4])
            if message[4]:
                print('episode: {}/{}, score: {}, e: {:.2}'.format(e, num_episodes, time, agent.epsilon))
                break
            if len(agent.replay_buffer) > batch_size:
                agent.replay(batch_size)
