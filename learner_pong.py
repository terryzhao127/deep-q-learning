from data_pb2 import Data
import zmq
import os
import random
from collections import deque

import gym
import numpy as np
from tensorflow.keras.layers import Dense,Conv2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras import backend as K
import horovod.tensorflow.keras as hvd
import tensorboardX


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
        #model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.compile(loss='mse', optimizer=hvd.DistributedOptimizer(Adam(lr=self.learning_rate*hvd.size())))
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
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    callbacks=[
        hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    ]

    writer = tensorboardX.SummaryWriter()

    env = gym.make('Pong-ram-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    # agent.load('./save/cartpole-dqn.h5')

    done = False
    batch_size = 32
    num_episodes = 1000000
    start_steps=10000
    PIPELINE=10

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind("tcp://*:5557")
    os.environ['KMP_WARNINGS']='off'
    id

    if not os.path.exists('save'):
        os.makedirs('save')

    sum=0
    all_rewards=[]
    for e in range(num_episodes):
        #state = env.reset()
        #state = np.reshape(state, [1, state_size])
        score=0
        for time in range(100000):
            # env.render()
            id,message0=socket.recv_multipart()
            message=Data()
            message.ParseFromString(message0)
            action=message.action
            reward=message.reward
            done=message.done
            state=np.zeros([1,state_size],dtype=float)
            next_state=np.zeros([1,state_size],dtype=float)
            for i in range(state_size):
                state[0][i]=message.state[i].element
                next_state[0][i]=message.next_state[i].element
            agent.memorize(state,action,reward,next_state,done)
            #file.write(str((state,action,reward,next_state,done))+'\n')
            #state = next_state
            #socket.send(b"1")
            score+=reward
            if done:
                #print(time)
                all_rewards.append(score)
                mean_reward=np.mean(all_rewards[-100:])
                sum=sum+reward
                print('episode: {}/{}, mean: {}, score: {}, e: {:.2}'.format(e, num_episodes, mean_reward, score, agent.epsilon))
                break
            if len(agent.replay_buffer) > batch_size:
                agent.replay(batch_size)
        sum+=score
        if e % 5 == 0 and hvd.rank()==0:
            agent.save('./save/cartpole-dqn-{}.h5'.format(e))
            file=open('./save/cartpole-dqn-{}.h5'.format(e),'rb')
            socket.send_multipart([id,file.read()])
            print('model sent')
            file.close()
            if e % 100 ==0 :
                writer.add_scalar("cartpole score",sum/100,e)
                writer.add_scalar("cartpole score",mean_reward,e)
                sum=0

    writer.close()