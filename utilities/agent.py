import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from utilities.replay_buffer import ReplayBuffer


def define_model(state_size, action_size):
    model = Sequential()
    model.add(Conv2D(16, 8, 4, activation='relu', input_shape=state_size))
    model.add(Conv2D(32, 4, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    return model


class Actor:
    def __init__(self, state_size, action_size, save_dir='actor_save'):
        self.model = define_model(state_size, action_size)
        self.action_size = action_size
        self.time_steps = 100000
        self.epsilon = 1.0
        self.epsilon_min = 0.01

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            act_values = self.model.predict(state[np.newaxis])
            return np.argmax(act_values[0])

    def load(self, weight, filename):
        filename = '{}/{}'.format(self.save_dir, filename)
        with open(filename, 'wb') as f:
            f.write(weight)
        self.model.load_weights(filename)

    def adjust_ep(self, step):
        fraction = min(1.0, float(step) / self.time_steps)
        self.epsilon = 1 + fraction * (self.epsilon_min - 1)


class Learner:
    def __init__(self, state_size, action_size, save_dir='learner_save'):
        self.gamma = 0.99  # Discount Rate
        self.memory = ReplayBuffer(5000)

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.policy_model = define_model(state_size, action_size)
        self.target_model = define_model(state_size, action_size)
        self.update_target_model('model.h5')

    def update_target_model(self, name):
        self.policy_model.save_weights('{}/{}'.format(self.save_dir, name))
        self.target_model.load_weights('{}/{}'.format(self.save_dir, name))

    def replay(self, batch_size, callbacks=None):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        next_action = np.argmax(self.policy_model.predict(next_states), axis=-1)
        target = rewards + (1-dones) * self.gamma * self.target_model.predict(next_states)[np.arange(batch_size), next_action]
        target_f = self.policy_model.predict(states)
        target_f[np.arange(batch_size), actions] = target
        self.policy_model.fit(states, target_f, epochs=1, verbose=1, callbacks=callbacks)




