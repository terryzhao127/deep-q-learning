dealer = ctx.socket(zmq.DEALER)
dealer.connect("tcp://127.0.0.1:6000")
dealer.send(b"fetch")

total = 0       # Total bytes received
chunks = 0      # Total chunks received

while True:
    try:
        chunk = dealer.recv()
    except zmq.ZMQError as e:
        if e.errno == zmq.ETERM:
            return   # shutting down, quit
        else:
            raise

    chunks += 1
    size = len(chunk)
    total += size
    if size == 0:
        break   # whole file received

print ("%i chunks received, %i bytes" % (chunks, total))
pipe.send(b"OK")