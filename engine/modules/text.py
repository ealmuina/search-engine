import argparse
import json
import socketserver

from engine.modules.utils import receive_json


def process(data):
    # TODO Do some actual work
    result = {
        'terms': data.lower().split()
    }
    return json.dumps(result)


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        request = receive_json(self.request)

        if request['action'] == 'process':
            self.request.sendall(process(request['data']).encode())
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

    server = socketserver.TCPServer((NETWORK['text']['host'], NETWORK['text']['port']), TCPHandler)
    server.serve_forever()
