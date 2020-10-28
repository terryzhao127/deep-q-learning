import os
import numpy as np
from gym.wrappers import Monitor
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import RMSprop

from utilities.environment import get_env
from utilities.agent import define_model
from utilities.replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, state_size, action_size, save_dir='agent_save'):
        self.action_size = action_size
        self.time_steps = 100000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount Rate
        self.memory = ReplayBuffer(5000)

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.policy_model = define_model(state_size, action_size)
        self.target_model = define_model(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            act_values = self.policy_model.predict(state[np.newaxis])
            return np.argmax(act_values[0])

    def adjust_ep(self, step):
        fraction = min(1.0, float(step) / self.time_steps)
        self.epsilon = 1 + fraction * (self.epsilon_min - 1)

    def update_target_model(self, name):
        self.policy_model.save_weights('{}/{}'.format(self.save_dir, name))
        self.target_model.load_weights('{}/{}'.format(self.save_dir, name))

    def replay(self, batch_size):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        next_action = np.argmax(self.policy_model.predict(next_states), axis=-1)
        target = rewards + (1-dones) * self.gamma * self.target_model.predict(next_states)[np.arange(batch_size), next_action]
        target_f = self.policy_model.predict(states)
        target_f[np.arange(batch_size), actions] = target
        self.policy_model.fit(states, target_f, epochs=1, verbose=0)


if __name__ == '__main__':
    writer = SummaryWriter('logs')

    env = get_env('PongNoFrameskip-v4', 4)
    env = Monitor(env, 'video', video_callable=lambda ep: ep % 50 == 0, force=True)

    action_size = env.action_space.n
    state_size = env.observation_space.shape
    agent = Agent(state_size, action_size)
    agent.update_target_model('model-base.h5')
    agent.policy_model.compile(loss='huber_loss', optimizer=RMSprop(learning_rate=0.0001))

    batch_size = 32
    num_steps = 1000000
    start_steps = 10000
    update_freq = 100
    episode_rewards = [0.0]

    state = env.reset()
    for step in range(num_steps):

        agent.adjust_ep(step)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.memory.add(state, action, reward, next_state, done)

        state = next_state
        episode_rewards[-1] += reward

        if step > start_steps:
            agent.replay(batch_size)
            if step % update_freq == 0:
                agent.update_target_model('model-{}.h5'.format(step))

        if done:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 2)

            print('episode: {}, step: {}/{}, mean reward: {}, epsilon: {:.3f}'.format(
                num_episodes, step+1, num_steps, mean_100ep_reward, agent.epsilon))
            writer.add_scalar('reward', mean_100ep_reward, num_episodes)
            writer.add_scalar('epsilon', agent.epsilon, num_episodes)
            writer.flush()

            state = env.reset()
            episode_rewards.append(0.0)
