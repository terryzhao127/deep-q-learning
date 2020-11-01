from data_pb2 import Data
import zmq
import os
import random
from collections import deque

import gym
import numpy as np
from tensorflow.keras.layers import Dense,Conv2D
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
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    env = gym.make('Pong-ram-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    #agent.load('./save/cartpole-dqn.h5')

    done = False
    batch_size = 32
    num_episodes = 1000000
    PIPELINE=10

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect("tcp://172.17.0.10:5557")
    #socket.connect("tcp://localhost:5557")
    os.environ['KMP_WARNINGS']='off'

    if not os.path.exists('save'):
        os.makedirs('save')

    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for _ in range(random.randint(1, 30)):
            state, _, _, _ = env.step(0)
            state = np.reshape(state, [1, state_size])
        for time in range(100000):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            data=Data()
            data.action=action
            data.reward=reward
            data.done=done
            for i in range(state_size):
                temp1=data.state.add()
                temp1.element=state[0][i]
                temp2=data.next_state.add()
                temp2.element=next_state[0][i]
            socket.send(data.SerializeToString())
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            #message = socket.recv()
            #print(str(reward)+' '+str(done))
            if done:
                # print('episode: {}/{}, score: {}, e: {:.2}'.format(e, num_episodes, time, agent.epsilon))
                print("current episode: %d"%e)
                break
            if len(agent.replay_buffer) > batch_size:
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                #agent.replay(batch_size)
        if e % 5 == 0:
            parameters=socket.recv()
            file=open('./save/cartpole-dqn-{}.h5'.format(e),'wb')
            file.write(parameters)
            file.close()
            agent.load('./save/cartpole-dqn-{}.h5'.format(e))
