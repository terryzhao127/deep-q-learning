import random
import zmq
from collections import deque

import gym
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
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    def replay(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            #训练网络
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
    done = False#游戏结束标志
    batch_size = 32
    #num_episodes = 1000

    counter = 1
    version = 1
    while True:

        #  Wait for next request from client
        message = socket.recv()
        sendb = b''
        #print("Received request: %s" % message)
        agent.memorize(np.array(eval(message)[0]), eval(message)[1], eval(message)[2], np.array(eval(message)[3]), eval(message)[4])

        #更新模型
        if len(agent.replay_buffer) > batch_size:
            agent.replay(batch_size)
            print("train epoch: ",counter)
            if counter % 3 == 0:
                agent.save('ver_pra{}.h5'.format(version))
                fo = open('ver_pra{}.h5'.format(version), 'rb')
                sendb = fo.read()
                fo.close()
                print('ver_pra{}.h5 saved'.format(version))
                version += 1
            counter += 1
            
        #  Send reply back to client
        socket.send(sendb)
        
        