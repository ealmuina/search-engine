import json
import socket

from django.conf import settings


def build(path):
    send_to_models(json.dumps({
        'action': 'build',
        'path': path
    }))


def search(query, count):
    response = send_to_models(json.dumps({
        'action': 'query',
        'query': query,
        'count': count
    }), True)
    return json.loads(response)


def send_to_models(request, wait_response=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to models module server and send request
        sock.connect((settings.NETWORK['models']['host'], settings.NETWORK['models']['port']))
        sock.sendall(request.encode())

        if wait_response:
            # Receive data from the server and shut down
            return sock.recv(1024).decode()
