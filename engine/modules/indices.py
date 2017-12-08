import argparse
import json
import os
import socketserver
import time

from engine.modules.utils import receive_json


class Index:
    def __init__(self, path):
        self.storage = {}
        self.path = os.path.join(path, 'index.json')
        if os.path.exists(self.path):
            self._load()

    def _load(self):
        with open(self.path) as index:
            self.storage = json.load(index)

    def _save(self):
        with open(self.path, 'w') as index:
            json.dump(self.storage, index)

    def create(self, data):
        for entry in data:
            self.storage[entry['key']] = entry['value']
        self._save()

    def delete(self, key):
        self.storage.pop(key, None)
        self._save()

    def get(self, key):
        value = self.storage.get(key, {})
        return json.dumps(value)

    def update(self, key, value):
        self.storage[key] = value
        self._save()


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global INDEX
        request = receive_json(self.request)
        start = time.time()

        if request['action'] == 'load':
            INDEX = Index(request['path'])
            self.request.sendall(json.dumps(INDEX.storage).encode())
        elif request['action'] == 'create':
            INDEX.create(request['data'])
        elif request['action'] == 'add':
            INDEX.update(request['key'], request['value'])
        elif request['action'] == 'update':
            INDEX.update(request['key'], request['value'])
        elif request['action'] == 'delete':
            INDEX.delete(request['key'])
        elif request['action'] == 'get':
            self.request.sendall(INDEX.get(request['key']).encode())
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
    INDEX = Index('')

    server = socketserver.TCPServer((NETWORK['indices']['host'], NETWORK['indices']['port']), TCPHandler)
    server.serve_forever()
