import json
import os
import socket
import subprocess

RESERVED_FILES = {
    'index.json', 'suggestions.bin'
}


def fix_pdf(path):
    name = os.path.basename(path)
    if not os.path.exists(name[:-4] + '.txt'):
        subprocess.run(['pdftotext', path])


def receive_json(sock):
    result = ''
    while True:
        current = sock.recv(4096)
        if not current:
            return json.loads(result)
        result += current.decode()


def send_json(dic, host, port, wait_response=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send request
        sock.connect((host, port))
        sock.sendall(json.dumps(dic).encode())

        if wait_response:
            sock.shutdown(socket.SHUT_WR)
            return receive_json(sock)
