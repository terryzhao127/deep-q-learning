import zmq
import numpy as np
from gym.wrappers import Monitor
from tensorboardX import SummaryWriter
from data_pb2 import Data
from env_wrappers import make_env
from DQNagent import Actor


if __name__ == '__main__':

    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.17.0.9:5555")

    writer = SummaryWriter('logs')

    env = make_env('PongNoFrameskip-v4', 4)
    env = Monitor(env, 'video', video_callable=lambda ep: ep % 50 == 0, force=True)
    action_size = env.action_space.n
    state_size = env.observation_space.shape
    actor = Actor(state_size, action_size)

    done = False
    num_episodes = 1000000
    cover_num = 0
    episode_rewards = [0.0]

    state = env.reset()
    for e in range(num_episodes):
        actor.adjust_ep(e)
        action = actor.act(state)
        next_state, reward, done, inf = env.step(action)
        
        message = Data(state=str(state.tolist()), next_state=str(next_state.tolist()), action=int(action),reward=reward, done=done)
        socket.send(message.SerializeToString())
        message = socket.recv()
        if message == b'Cover':
            cover_num += 1
            if cover_num == 100:
                print('Receive 100 message')
        else:
                print('Cover new model')
                actor.load(message, 'model.h5')

        state = next_state
        episode_rewards[-1] += reward
        print('episode: {}, reward:{}, done:{}'.format(e, reward, done))
        if done:
            count_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 2)

            #print('episode: {}, step: {}/{}, mean reward: {}, epsilon: {:.3f}'.format(
            #    count_episodes, e+1, num_episodes, mean_100ep_reward, actor.epsilon))
            #writer.add_scalar('reward', mean_100ep_reward, count_episodes)
            #writer.add_scalar('epsilon', actor.epsilon, count_episodes)
            #writer.flush()

            state = env.reset()
            episode_rewards.append(0.0)

