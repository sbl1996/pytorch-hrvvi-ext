import io

import zmq

import torch


def dumps(t):
    s = io.BytesIO()
    torch.save(t, s)
    return s.getvalue()


def loads(s):
    s = io.BytesIO(initial_bytes=s)
    t = torch.load(s)
    return t


def new_consumer():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    port = socket.bind_to_random_port("tcp://*")
    return context, socket, port


def new_producer(port):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f'tcp://localhost:{port}')
    return socket


def put(socket, data):
    data = dumps(data)
    socket.send(data)


def get(socket):
    data = socket.recv()
    data = loads(data)
    return data
