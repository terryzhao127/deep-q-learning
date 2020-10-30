import random
from collections import deque
from data_pb2 import Data
import gym
import zmq
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop
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
    
    # horovod init
    htk.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(htk.local_rank())
    K.set_session(tf.Session(config=config))
    callbacks = [htk.callbacks.BroadcastGlobalVariablesCallback(0)]
    
    # zmq init
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    # env init
    env = gym.make('PongNoFrameskip-v4', 4)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    # agent.load('./save/cartpole-dqn.h5')
    opt = htk.DistributedOptimizer(RMSprop(learning_rate=0.0001 * htk.size()))
    agent.policy_model.compile(loss='huber_loss', optimizer=opt)

    batch_size = 32 // htk.size()
    num_steps = 1000000 // htk.size()
    start_steps = 10000 // htk.size()
    update_freq = 1000 // htk.size()
    newmodel = 0

    for e in range(num_steps):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
    
        message = Data()
        message.ParseFromString(socket.recv())
        if newmodel:
            socket.send(newmodel)
        else:
            socket.send(b"Cover")

        state, next_state = eval(message.state), eval(message.next_state)
        agent.memorize(np.array(state), message.action, message.reward, np.array(next_state), message.done)
         
            
        if e % update_freq == 0:
            agent.save('save/model.h5')
            newmodel = open('save/model.h5', 'rb').read()
        else:
            newmodel = 0

        if len(agent.replay_buffer) > batch_size:
            agent.replay(batch_size, callbacks)
