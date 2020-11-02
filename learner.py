import zmq
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from data_pb2 import Data
from env_wrappers import make_env
from DQNagent import Learner
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K


if __name__ == '__main__':
    # horovod init
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    
    # zmq init
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    # env init
    env = make_env('PongNoFrameskip-v4', 4)
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    learner = Learner(state_size, action_size, 'learner_{}'.format(hvd.rank()))
    learner.update_target_model('model.h5')
    opt = hvd.DistributedOptimizer(RMSprop(learning_rate=0.0001 * hvd.size()))
    learner.policy_model.compile(loss='huber_loss', optimizer=opt)

    batch_size = 32 // hvd.size()
    num_steps = 1000000 // hvd.size()
    start_steps = 10000 // hvd.size()
    update_freq = 1000 // hvd.size()
    newmodel = 0

    for e in range(num_steps):
    
        message = Data()
        message.ParseFromString(socket.recv())
        if newmodel:
            socket.send(newmodel)
        else:
            socket.send(b"Cover")
        state, next_state = eval(message.state), eval(message.next_state)
        learner.memorize(np.array(state), message.action, message.reward, np.array(next_state), message.done)
            
        if len(learner.memory) > batch_size:
            learner.replay(batch_size, callbacks)

        if e % update_freq == 0:
            learner.save('save/model.h5')
            newmodel = open('save/model.h5', 'rb').read()
        else:
            newmodel = 0

