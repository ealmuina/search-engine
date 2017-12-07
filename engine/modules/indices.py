import argparse
import json
import os
import socketserver

from engine.modules.utils import receive_string


class Index:
    def __init__(self, path):
        self.storage = {}
        self.path = os.path.join(path, 'index.json')

    def _load(self):
        with open(self.path) as index:
            self.storage = json.load(index)

    def _save(self):
        with open(self.path, 'w') as index:
            json.dump(self.storage, index)

    def create(self, data):
        if os.path.exists(self.path):
            self._load()
        else:
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
    def __init__(self, *args, **kwargs):
        self.index = None
        super().__init__(*args, **kwargs)

    def handle(self):
        data = receive_string(self.request)
        request = json.loads(data)

        if request['action'] == 'create':
            self.index = Index(request.get('path', ''))
            self.index.create(request['data'])
        elif not self.index:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'A "create" action must be issued before any other operation with the indices module.'
            }).encode())
        elif request['action'] == 'add':
            self.index.update(request['key'], request['value'])
        elif request['action'] == 'update':
            self.index.update(request['key'], request['value'])
        elif request['action'] == 'delete':
            self.index.delete(request['key'])
        elif request['action'] == 'get':
            self.request.sendall(self.index.get(request['key']).encode())
        else:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Invalid action.'
            }).encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('network')
    args = parser.parse_args()

    NETWORK = json.load(open(args.network))

    server = socketserver.TCPServer((NETWORK['indices']['host'], NETWORK['indices']['port']), TCPHandler)
    server.serve_forever()
