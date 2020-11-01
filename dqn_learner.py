import random
import zmq
import os
import json
import experience_pb2
import tensorboardX
from collections import deque


import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import horovod.tensorflow.keras as hvd
from tensorflow.keras import backend as K


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
        model.compile(loss='mse', optimizer=hvd.DistributedOptimizer(Adam(lr=self.learning_rate * hvd.size())))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

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
    env = gym.make('Pong-ram-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))
    hvd.callbacks.BroadcastGlobalVariablesCallback(0)

    agent = DQNAgent(state_size, action_size)
    os.environ['KMP_WARNINGS'] = '0'
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
    socket = zmq.Context().socket(zmq.ROUTER)
    socket.bind("tcp://*:6080")

    model_path = "./model_weights"
    if hvd.rank() == 0 and os.path.exists(model_path):
        os.system("rm -rf " + model_path)
    os.mkdir(model_path)

    writer = tensorboardX.SummaryWriter()

    batch_size = 32
    num_episodes = 100000
    total_reward = []
    sum = 0
    for e in range(num_episodes):
        score = 0
        for time in range(100000):
            id, raw_data = socket.recv_multipart()
            data = experience_pb2.Exper()
            data.ParseFromString(raw_data)
            data = {
                "now_state": [data.now_state.pos],
                "action": data.action,
                "reward": data.reward,
                "next_state": [data.next_state.pos],
                "done": data.done,
            }

            agent.memorize(np.array(data["now_state"]), data["action"], data["reward"], np.array(data["next_state"]), data["done"])
            reward = data['reward']
            score += reward
            if data["done"]:
                total_reward.append(reward)
                if len(total_reward) > 100:
                    total_reward = total_reward[-100:]
                print('learner -- episode: {}/{}, score: {}, mean_reward: {}, e: {:.2}'.format(e, num_episodes, score, np.mean(total_reward), agent.epsilon))
                break
            if len(agent.replay_buffer) > batch_size:
                agent.replay(batch_size)
                # agent.save(model_path + "/syf-eposide_{}-time_{}.h5".format(e, time))
                # with open(model_path + "/syf-eposide_{}-time_{}.h5".format(e, time), "rb") as file:
                #     socket.send_multipart([id, file.read()])
        sum += score
        if e % 3 == 0 and hvd.rank() == 0:
            print("learner -- send model")
            agent.save(model_path + "/syf-eposide_{}.h5".format(e, time))
            with open(model_path + "/syf-eposide_{}.h5".format(e, time), "rb") as file:
                socket.send_multipart([id, file.read()])
            writer.add_scalar("cartpole score",sum / 100, e)
            writer.add_scalar("cartpole score",np.mean(total_reward), e)
            sum = 0

    writer.close()
