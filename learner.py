import zmq
from tensorflow.keras.optimizers import RMSprop

from utilities.environment import get_env
from utilities.agent import Learner
from utilities.data import Data, bytes2arr

import tensorflow as tf
from tensorflow.keras import backend as K
import horovod.tensorflow.keras as hvd


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]


if __name__ == '__main__':
    env = get_env('PongNoFrameskip-v4', 4)
    action_size = env.action_space.n
    state_size = env.observation_space.shape

    learner = Learner(state_size, action_size, 'learner_{}'.format(hvd.rank()))
    learner.update_target_model('model.h5')

    opt = hvd.DistributedOptimizer(RMSprop(learning_rate=0.0001 * hvd.size()))
    learner.policy_model.compile(loss='huber_loss', optimizer=opt)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://172.17.0.12:6659")

    batch_size = 32 // hvd.size()
    num_steps = 1000000 // hvd.size()
    start_steps = 10000 // hvd.size()
    update_freq = 1000 // hvd.size()

    weight = b''
    for step in range(num_steps):

        socket.send(weight)
        weight = b''

        data = Data()
        data.ParseFromString(socket.recv())
        state, next_state = bytes2arr(data.state), bytes2arr(data.next_state)
        learner.memory.add(state, data.action, data.reward, next_state, data.done)

        if step > start_steps:
            learner.replay(batch_size, callbacks)

            if step % update_freq == 0:
                learner.update_target_model('model.h5'.format(step))

            if hvd.rank() == 0:
                learner.policy_model.save_weights('{}/{}'.format(learner.save_dir, 'model.h5'))
                with open('{}/model.h5'.format(learner.save_dir), 'rb') as f:
                    weight = f.read()
