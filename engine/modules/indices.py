import argparse
import json
import socketserver

from engine.models import Term, Occurrence


class VectorIndex:
    @staticmethod
    def _add_occurrences(term, value):
        bulk = []
        for occurrence in value['occurrences']:
            bulk.append(Occurrence(document_id=occurrence['document'], term=term, weight=occurrence['weight']))
        Occurrence.objects.bulk_create(bulk)

    @staticmethod
    def add(key, value):
        term = Term(name=key)
        term.k = value['k']
        term.save()
        VectorIndex._add_occurrences(term, value)

    @staticmethod
    def create(data):
        bulk = []
        for term in data:
            bulk.append(Term.objects.create(name=term['key'], k=term['value']['k']))
        Term.objects.bulk_create(bulk)
        for i in range(len(data)):
            VectorIndex._add_occurrences(bulk[i], data[i]['value'])

    @staticmethod
    def delete(key):
        term = Term.objects.get(name=key)
        term.delete()

    @staticmethod
    def get(key):
        try:
            term = Term.objects.get(name=key)
            return json.dumps({
                'success': True,
                'value': {
                    'k': term.k,
                    'occurrences': [{
                        'document': occurrence.document,
                        'weight': occurrence.weight
                    } for occurrence in term.occurrence_set]
                }
            })
        except Term.DoesNotExist:
            return json.dumps({
                'success': False
            })

    @staticmethod
    def update(key, value):
        term = Term.objects.get(name=key)
        term.k = value['k']
        term.save()
        term.occurrence_set.all().delete()
        VectorIndex._add_occurrences(term, value)


class TCPHandler(socketserver.BaseRequestHandler):
    def __init__(self, *args, **kwargs):
        self.index = None
        super().__init__(*args, **kwargs)

    def handle(self):
        data = self.request.recv(1024).decode()
        request = json.loads(data)

        if request['action'] == 'create':
            VectorIndex.create(request['data'])
        elif not self.index:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Uninitialized index. Please request a "create" action before any other operation.'
            }).encode())
        elif request['action'] == 'add':
            VectorIndex.add(request['key'], request['value'])
        elif request['action'] == 'update':
            VectorIndex.update(request['key'], request['value'])
        elif request['action'] == 'delete':
            VectorIndex.delete(request['key'])
        elif request['action'] == 'get':
            self.request.sendall(VectorIndex.get(request['key']).encode())
        else:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Invalid action.'
            }).encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='VectorIndex')
    args = parser.parse_args()

    INDEX = {
        'VectorIndex': VectorIndex,
    }[args.model]

    server = socketserver.TCPServer(('0.0.0.0', 9904), TCPHandler)
    server.serve_forever()
