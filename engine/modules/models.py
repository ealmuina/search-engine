import argparse
from bisect import bisect_left
from functools import reduce
import json
from pathlib import Path
import socketserver
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from engine.modules.utils import send_json, receive_json


def _analyze(document):
    response = send_json({
        'action': 'process',
        'data': document.read()
    }, NETWORK['text']['host'], NETWORK['text']['port'], True)
    return response['terms']


class Vector:
    def __init__(self):
        self.path = ''
        self.w = None
        self.terms = None
        self.term_count = 0
        self.doc_count = 0
        self.doc_names = None

    def _build_index(self, freq, terms):
        index = [{'key': term, 'value': {'documents': []}} for term in terms]
        freq = np.array(freq.todense())

        it = np.nditer(freq, flags=['multi_index'])
        while not it.finished:
            f = int(it[0])
            i, j = it.multi_index
            if f:
                index[i]['value']['documents'].append({'document': self.doc_names[j], 'freq': f})
            it.iternext()

        send_json({
            'action': 'create',
            'data': index
        }, NETWORK['indices']['host'], NETWORK['indices']['port'])

    def _calculate_freq(self):
        vectorizer = CountVectorizer(input='file', analyzer=_analyze)
        documents = []
        for doc in Path(self.path).iterdir():
            if doc.name == 'index.json':
                continue
            documents.append(open(str(doc)))

        freq = vectorizer.fit_transform(documents).transpose()
        terms = vectorizer.get_feature_names()
        return freq, terms

    def _load_freq(self, index):
        terms = sorted(list(index.keys()))
        documents = [doc.name for doc in Path(self.path).iterdir() if doc.name != 'index.json']
        freq = np.zeros((len(terms), len(documents)))

        for t in terms:
            i = bisect_left(terms, t)
            for d in index[t]['documents']:
                j = bisect_left(documents, d['document'])
                freq[i, j] = d['freq']

        return freq, terms

    def _get_similarities(self, q):
        vectorizer = TfidfVectorizer(vocabulary=self.terms)
        q = vectorizer.fit_transform([q])
        similarities = cosine_similarity(q, self.w.transpose()).tolist()[0]
        return similarities

    def build(self, path):
        self.path = path
        self.doc_names = [doc.name for doc in Path(self.path).iterdir() if doc.name != 'index.json']

        index = send_json({
            'action': 'load',
            'path': path
        }, NETWORK['indices']['host'], NETWORK['indices']['port'], True)

        if not index:
            freq, terms = self._calculate_freq()
            self._build_index(freq, terms)
        else:
            freq, terms = self._load_freq(index)

        self.w = TfidfTransformer().fit_transform(freq)
        self.terms = terms
        self.term_count = self.w.shape[0]
        self.doc_count = self.w.shape[1]

    def query(self, q, count):
        similarities = self._get_similarities(q)
        documents = np.argsort(similarities)[-count:][::-1]
        documents = [doc for doc in documents if similarities[doc] > 0]

        if documents:
            result = {
                'action': 'report',
                'success': True,
                'results': [
                    {
                        'document': self.doc_names[doc],
                        'match': similarities[doc]
                    } for doc in documents
                ]
            }
        else:
            result = {
                'action': 'report',
                'success': False
            }
        return json.dumps(result)


class GeneralizedVector(Vector):
    def __init__(self):
        super().__init__()
        self.k = None

    def _calculate_wong_k(self):
        w = np.array(self.w.todense()).tolist()
        minterms = []
        for j in range(self.doc_count):
            m = 0
            for i in range(self.term_count):
                m += 2 ** i if w[i][j] else 0
            minterms.append(m)

        # Calculate correlations
        m = sorted(list(set(minterms)))
        c = np.zeros((self.term_count, len(m))).tolist()
        for i in range(self.term_count):
            for j in range(self.doc_count):
                r = bisect_left(m, minterms[j])
                c[i][r] += w[i][j]

        # Calculate the index term vectors as linear combinations of minterm vectors
        k = []
        for i in range(self.term_count):
            num = reduce(
                lambda acum, r: acum + np.array([c[i][r] if 2 ** l & m[r] else 0 for l in range(self.term_count)]),
                range(len(m)),
                np.zeros(self.term_count)
            )
            k.append(num / np.linalg.norm(num))

        return np.array(k)

    def _calculate_pearson_k(self):
        w = np.array(self.w.todense())
        k = np.corrcoef(w)
        return np.abs(k)

    def _get_similarities(self, q):
        vectorizer = TfidfVectorizer(vocabulary=self.terms)
        q = vectorizer.fit_transform([q])

        q = q.dot(self.k)
        d = self.w.transpose().dot(self.k)

        similarities = cosine_similarity(q, d).tolist()[0]
        return similarities

    def build(self, path):
        super().build(path)
        self.k = self._calculate_pearson_k()


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        request = receive_json(self.request)
        start = time.time()

        if request['action'] == 'build':
            MODEL.build(request['path'])
        elif request['action'] == 'query':
            self.request.sendall(MODEL.query(request['query'], request['count']).encode())
        else:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Invalid action.'
            }).encode())

        print('Processed action "%s" in %.2f seconds' % (request['action'], time.time() - start))


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
    }[args.model]()
    NETWORK = json.load(open(args.network))

    server = socketserver.TCPServer((NETWORK['models']['host'], NETWORK['models']['port']), TCPHandler)
    server.serve_forever()
