import zmq
from zhelpers import socket_set_hwm

ctx = zmq.Context()
CHUNK_SIZE = 250000
file = open("testdata", "rb")
router = ctx.socket(zmq.ROUTER)

# Default HWM is 1000, which will drop messages here
# since we send more than 1,000 chunks of test data,
# so set an infinite HWM as a simple, stupid solution:
socket_set_hwm(router, 0)
router.bind("tcp://*:6000")

while True:
    # First frame in each message is the sender identity
    # Second frame is "fetch" command
    try:
        identity, command = router.recv_multipart()
    except zmq.ZMQError as e:
        if e.errno == zmq.ETERM:
            return   # shutting down, quit
        else:
            raise

    assert command == b"fetch"

    while True:
        data = file.read(CHUNK_SIZE)
        router.send_multipart([identity, data])
        if not data:
            break