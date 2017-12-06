import argparse
from bisect import bisect_left
from functools import reduce
import json
import socketserver

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class _Model:
    def __init__(self):
        pass

    def _similarity(self, j, q):
        raise NotImplementedError('This method has to be implemented by inheritors')

    def build(self, path):
        pass

    def query(self, q, count):
        similarities = list(map(lambda j: self._similarity(j, q), range(self.doc_count)))
        similarities = [(similarities[i], i) for i in range(len(similarities)) if similarities[i] > 0]
        similarities.sort(reverse=True)
        similarities = similarities[:count]

        if similarities:
            result = {
                'action': 'report',
                'success': True,
                'results': [
                    {
                        'document': index,
                        'match': similarity
                    } for similarity, index in similarities
                ]
            }
        else:
            result = {
                'action': 'report',
                'success': False
            }
        return json.dumps(result)


class Vector(_Model):
    def __init__(self, d):
        super().__init__()
        vectorizer = TfidfVectorizer()
        self.w = vectorizer.fit_transform(d).transpose()
        self.terms = vectorizer.get_feature_names()
        self.term_count = self.w.shape[0]
        self.doc_count = self.w.shape[1]

    def _similarity(self, j, q):
        vectorizer = TfidfVectorizer(vocabulary=self.terms)
        q = vectorizer.fit_transform([q]).transpose()
        num = sum([self.w[i, j] * q[i] for i in range(self.term_count)])[0, 0]
        den = np.math.sqrt(reduce(lambda x, y: x + y ** 2, self.w[:, j], 0)[0, 0]) * np.math.sqrt(
            reduce(lambda x, y: x + y ** 2, q, 0)[0, 0])
        return num / den

    def build(self, path):
        pass


class GeneralizedVector(_Model):
    def __init__(self, d):
        super().__init__()
        vectorizer = TfidfVectorizer()
        self.w = vectorizer.fit_transform(d).transpose()
        self.terms = vectorizer.get_feature_names()
        self.term_count = self.w.shape[0]
        self.doc_count = self.w.shape[1]

        # Calculate minterms
        minterms = []
        for j in range(self.doc_count):
            m = 0
            for i in range(self.term_count):
                m += 2 ** i if self.w[i, j] else 0
            minterms.append(m)

        # Calculate correlations
        m = sorted(list(set(minterms)))
        c = np.zeros((self.term_count, len(m)))
        for i in range(self.term_count):
            for j in range(self.doc_count):
                r = bisect_left(m, minterms[j])
                c[i, r] += self.w[i, j]

        # Calculate the index term vectors as linear combinations of minterm vectors
        self.k = np.zeros((self.term_count, self.term_count))
        for i in range(self.term_count):
            num = reduce(
                lambda acum, r: acum + np.array([c[i, r] if 2 ** l & m[r] else 0 for l in range(self.term_count)]),
                range(len(m)),
                np.zeros(self.term_count)
            )
            self.k[i] = num / np.linalg.norm(num)

    def _similarity(self, j, q):
        vectorizer = TfidfVectorizer(vocabulary=self.terms)
        w = vectorizer.fit_transform([q]).transpose()
        q = reduce(
            lambda acum, i: acum + w[i, 0] * self.k[i],
            range(self.term_count),
            np.zeros(self.term_count)
        )
        d = reduce(lambda acum, i: acum + self.w[i, j] * self.k[i], range(self.term_count), np.zeros(self.term_count))

        return cosine_similarity(q.reshape(1, -1), d.reshape(1, -1))[0, 0]

    def build(self, path):
        pass


class TCPHandler(socketserver.BaseRequestHandler):
    def __init__(self, *args, **kwargs):
        self.model = None
        super().__init__(*args, **kwargs)

    def handle(self):
        data = self.request.recv(1024).decode()
        request = json.loads(data)

        if request['action'] == 'build':
            self.model = MODEL()
            self.model.build(request['path'])
        elif not self.model:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Uninitialized model. Please request a "build" action before any other operation.'
            }).encode())
        elif request['action'] == 'query':
            self.request.sendall(self.model.query(request['query'], request['count']).encode())
        else:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Invalid action.'
            }).encode())


def test():
    docs = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?'
    ]
    gv = GeneralizedVector(docs)
    v = Vector(docs)
    q = 'third'
    for j in range(4):
        print(gv._similarity(j, q))
    print()
    for j in range(4):
        print(v._similarity(j, q))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('network')
    parser.add_argument('--model', default='GeneralizedVector')
    args = parser.parse_args()

    MODEL = {
        'Vector': Vector,
        'GeneralizedVector': GeneralizedVector
    }[args.model]
    NETWORK = json.load(open(args.network))

    server = socketserver.TCPServer((NETWORK['models']['host'], NETWORK['models']['port']), TCPHandler)
    server.serve_forever()
