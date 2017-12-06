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
    server = socketserver.TCPServer(('0.0.0.0', 9901), TCPHandler)
    server.serve_forever()
