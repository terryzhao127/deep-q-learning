import zmq

ctx = zmq.Context()
CHUNK_SIZE = 250000
dealer = ctx.socket(zmq.DEALER)
dealer.connect("tcp://172.0.0.4:5555")
dealer.send(b"fetch")

total = 0       # Total bytes received
chunks = 0      # Total chunks received

while True:
    try:
        chunk = dealer.recv()
    except zmq.ZMQError as e:
        if e.errno == zmq.ETERM:
            return -1# shutting down, quit
        else:
            raise

    chunks += 1
    size = len(chunk)
    total += size
    if size == 0:
        break   # whole file received

print ("%i chunks received, %i bytes" % (chunks, total))
pipe.send(b"OK")