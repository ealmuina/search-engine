import json
import os
import pathlib
import socket
import subprocess
from bisect import bisect_left

import numpy as np

RESERVED_FILES = {
    'index.json', 'suggestions.bin', 'summary.bin'
}


def fix_pdf(path):
    name = os.path.basename(path)
    if not os.path.exists(name[:-4] + '.txt'):
        subprocess.run(['pdftotext', path])


def get_documents(path):
    path_docs = set()
    for doc in pathlib.Path(path).iterdir():
        if doc.name in RESERVED_FILES:
            continue
        path_docs.add(doc.name[:-4])
    return path_docs


def load_freq(index, doc_names):
    terms = sorted(list(index.keys()))
    freq = np.zeros((len(terms), len(doc_names)))

    for t in terms:
        i = bisect_left(terms, t)
        for d in index[t]['documents']:
            j = bisect_left(doc_names, d['document'])
            freq[i, j] = d['freq']

    return freq, terms


def receive_json(sock):
    result = ''
    while True:
        current = sock.recv(4096)
        if not current:
            return json.loads(result)
        result += current.decode()


def remove_caches(path):
    for file in RESERVED_FILES:
        file_path = os.path.join(path, file)
        if os.path.exists(file_path):
            os.remove(file_path)


def send_json(dic, host, port, wait_response=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send request
        sock.connect((host, port))
        sock.sendall(json.dumps(dic).encode())

        if wait_response:
            sock.shutdown(socket.SHUT_WR)
            return receive_json(sock)
