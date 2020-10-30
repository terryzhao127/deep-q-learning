import random
from collections import deque
from data_pb2 import Data
import gym
import zmq
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import horovod.tensorflow.keras as htk

#htk.init()
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = str(htk.local_rank())
#K.set_session(tf.Session(config=config))

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
        self.model = self._build_model(state_size, action_size)

    def _build_model(self, state_size, action_size):
        """Build Neural Net for Deep Q-learning Model"""
        model = Sequential()
        model.add(Conv2D(16, 8, 4, activation='relu', input_shape=state_size))
        model.add(Conv2D(32, 4, 2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size, callbacks=None)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=1, callbacks=callbacks)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

callbacks = [htk.callbacks.BroadcastGlobalVariablesCallback(0)]

if __name__ == '__main__':
    
    htk.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(htk.local_rank())
    K.set_session(tf.Session(config=config))
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    # agent.load('./save/cartpole-dqn.h5')

    done = False
    batch_size = 32
    num_episodes = 1000
    cover_num = 0 
    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        send_flag = 0
        for time in range(500):
            
            #message = socket.recv()
            message = Data()
            message.ParseFromString(socket.recv())
            state, next_state = eval(message.state), eval(message.next_state)
            agent.memorize(np.array(state), message.action, message.reward, np.array(next_state), message.done)
            socket.send(b"Cover")
            #if e % 2 == 0:
            #    if send_flag == 0:
            #        agent.save('./save/cartpole-dqn{}.h5'.format(e))
            #        with open('save/cartpole-dqn{}.h5'.format(e), 'rb') as f:
            #            msend = f.read()
            #        socket.send(msend)
            #        send_flag = 1
            #    else:
            #        socket.send(b"Cover")
            #else:
            #    socket.send(b"Cover")
            
            #message = eval(message)
            #agent.memorize(np.array(message[0]), message[1], message[2], np.array(message[3]), message[4])
        
            #socket.send(b"Cover")
            if message.done:
                print('episode: {}/{}, score: {}, e: {:.2}'.format(e, num_episodes, time, agent.epsilon))
                break
            # env.render()
            #action = agent.act(state)
            #next_state, reward, done, _ = env.step(action)
            #reward = reward if not done else -10
            #next_state = np.reshape(next_state, [1, state_size])
            #agent.memorize(state, action, reward, next_state, done)
           
            #message = socket.recv()
            #if message == "Cover":
            #    cover_num += 1
            #if cover_num == 500:
            #    print("Send 500 message")

            #state = next_state
            #if done:
            #    print('episode: {}/{}, score: {}, e: {:.2}'.format(e, num_episodes, time, agent.epsilon))
            #    break
            
            if len(agent.replay_buffer) > batch_size:
                agent.replay(batch_size, callbacks)
