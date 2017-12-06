import argparse
import json
import socketserver


class TCPHandler(socketserver.BaseRequestHandler):
    def __init__(self, *args, **kwargs):
        self.index = None
        super().__init__(*args, **kwargs)

    def handle(self):
        data = self.request.recv(1024).decode()
        request = json.loads(data)

        if request['action'] == 'report':
            if request['success']:
                pass
            else:
                pass
        else:
            # TODO Show error message
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('network')
    args = parser.parse_args()

    NETWORK = json.load(open(args.network))

    server = socketserver.TCPServer((NETWORK['ui']['address'], NETWORK['ui']['port']), TCPHandler)
    server.serve_forever()
