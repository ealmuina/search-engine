import argparse
import json
import os
import pickle
import random
import socketserver
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np

import engine.modules.utils as utils


class Adviser:
    def __init__(self, path):
        self.path = path
        self.doc_names = [
            '.'.join(doc.name.split('.')[:-1])
            for doc in Path(self.path).iterdir() if doc.name.endswith('.txt')
        ]
        self.doc_names.sort()
        self.doc_count = len(self.doc_names)
        self.w = self._load_w()
        self.k = self._build_k()
        self.visits = {}

    def _build_k(self):
        index = self._load_index()
        freq, terms = utils.load_freq(index, self.doc_names)
        freq = freq.transpose()
        k = np.corrcoef(freq)
        return np.abs(k)

    def _load_index(self):
        while True:
            index = utils.send_json({
                'action': 'load',
                'path': self.path
            }, NETWORK['indices']['host'], NETWORK['indices']['port'], True)
            if index:
                break
            time.sleep(random.randint(10, 60))  # Give some time so indices module can make it
        return index

    def _load_w(self):
        path = os.path.join(self.path, 'suggestions.bin')
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
        else:
            return np.ones((self.doc_count, self.doc_count))

    def _save_w(self):
        path = os.path.join(self.path, 'suggestions.bin')
        with open(path, 'wb') as file:
            pickle.dump(self.w, file, pickle.HIGHEST_PROTOCOL)

    def suggest(self, token):
        if token in self.visits:
            usefulness = (self.w * self.k)[list(self.visits[token]), :]
            usefulness = np.max(usefulness, axis=0)
            documents = np.argsort(usefulness)[::-1]
            documents = [doc for doc in documents if usefulness[doc] > 0 and doc not in self.visits[token]]
        else:
            documents = usefulness = []

        if documents:
            result = {
                'action': 'report',
                'success': True,
                'results': [
                    {
                        'document': self.doc_names[doc],
                        'usefulness': usefulness[doc]
                    } for doc in documents
                ]
            }
        else:
            result = {
                'action': 'report',
                'success': False
            }
        return json.dumps(result)

    def fit(self, token, document):
        v = self.visits.get(token, set())
        i = bisect_left(self.doc_names, document)
        if i not in v:
            for j in v:
                self.w[i, j] += 1
                self.w[j, i] += 1
            v.add(i)
            self.visits[token] = v
            self._save_w()


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global ADVISER
        request = utils.receive_json(self.request)
        start = time.time()

        if request['action'] == 'build':
            success = True
            # noinspection PyBroadException
            try:
                ADVISER = Adviser(request['path'])
            except Exception:
                success = False
            self.request.sendall(json.dumps({
                'action': 'report',
                'success': success
            }).encode())
        elif not ADVISER:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': '"build" action required before any other action'
            }).encode())
        elif request['action'] == 'fit':
            ADVISER.fit(request['token'], request['document'])
        elif request['action'] == 'suggest':
            self.request.sendall(ADVISER.suggest(request['token']).encode())
        else:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Invalid action.'
            }).encode())

        print('Processed action "%s" in %.2f seconds' % (request['action'], time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('network')
    args = parser.parse_args()

    NETWORK = json.load(open(args.network))
    ADVISER = None

    server = socketserver.TCPServer((NETWORK['recommendation']['host'], NETWORK['recommendation']['port']), TCPHandler)
    server.serve_forever()
