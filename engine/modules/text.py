import json
import socketserver


def process(data):
    result = {
        'terms': []
    }
    return json.dumps(result)


class TCPHandler(socketserver.BaseRequestHandler):
    def __init__(self, *args, **kwargs):
        self.index = None
        super().__init__(*args, **kwargs)

    def handle(self):
        data = self.request.recv(1024).decode()
        request = json.loads(data)

        if request['action'] == 'process':
            self.request.sendall(process(request['data']).encode())
        else:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Invalid action.'
            }).encode())


if __name__ == '__main__':
    server = socketserver.TCPServer(('0.0.0.0', 9902), TCPHandler)
    server.serve_forever()
