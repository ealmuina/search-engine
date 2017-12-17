import argparse
import json
import os
import socketserver
import time
from bisect import bisect_left
from functools import reduce
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import engine.modules.utils as utils


def _analyze(text, is_query):
    response = utils.send_json({
        'action': 'process',
        'data': text,
        'is_query': is_query
    }, NETWORK['text']['host'], NETWORK['text']['port'], True)
    return response['terms']


def _analyze_query(query):
    return _analyze(query, True)


def _analyze_document(document):
    return _analyze(document.read(), False)


class Vector:
    def __init__(self):
        self.path = ''
        self.freq = None
        self.w = None
        self.terms = None
        self.term_count = 0
        self.doc_count = 0
        self.doc_names = None
        self.last_query = {}  # last query answered for each session

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

        utils.send_json({
            'action': 'create',
            'data': index
        }, NETWORK['indices']['host'], NETWORK['indices']['port'])

    def _calculate_freq(self):
        vectorizer = CountVectorizer(input='file', analyzer=_analyze_document)
        documents = []
        for doc in self.doc_names:
            path = os.path.join(self.path, doc + '.txt')
            documents.append(open(path))

        freq = vectorizer.fit_transform(documents).transpose()
        terms = vectorizer.get_feature_names()
        return freq, terms

    def _get_similarities(self, q):
        similarities = cosine_similarity(q, self.w.transpose()).tolist()[0]
        return similarities

    def build(self, path):
        self.path = path
        self.doc_names = [
            '.'.join(doc.name.split('.')[:-1])
            for doc in Path(self.path).iterdir() if doc.name.endswith('.txt')
        ]
        self.doc_names.sort()

        index = utils.send_json({
            'action': 'load',
            'path': path
        }, NETWORK['indices']['host'], NETWORK['indices']['port'], True)

        if not index:
            freq, terms = self._calculate_freq()
            self._build_index(freq, terms)
        else:
            freq, terms = utils.load_freq(index, self.doc_names)

        self.freq = freq
        self.w = TfidfTransformer().fit_transform(freq)
        self.terms = terms
        self.term_count = self.w.shape[0]
        self.doc_count = self.w.shape[1]

    def query(self, token, q, count):
        if count == -1:
            count = self.doc_count

        if isinstance(q, str):
            vectorizer = TfidfVectorizer(vocabulary=self.terms, analyzer=_analyze_query)
            q = vectorizer.fit_transform([q])

        self.last_query[token] = q
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

    def update_query(self, token, doc_name, positive, count):
        q = self.last_query[token]
        j = bisect_left(self.doc_names, doc_name)
        d = self.w[:, j].transpose()
        if positive:
            q += d
        else:
            q -= 0.5 * d
        return self.query(token, q, count)


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
        A = self.freq.astype(np.float64)
        n = A.shape[1]

        # Compute the covariance matrix
        rowsum = A.sum(1)
        centering = rowsum.dot(rowsum.T.conjugate()) / n
        C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

        # The correlation coefficients are given by
        # C_{i,j} / sqrt(C_{i} * C_{j})
        d = np.diag(C)
        k = C / np.sqrt(np.outer(d, d))

        return np.abs(k)

    def _get_similarities(self, q):
        q = q.dot(self.k)
        d = self.w.transpose().dot(self.k)

        similarities = cosine_similarity(q, d).tolist()[0]
        return similarities

    def build(self, path):
        super().build(path)
        self.k = self._calculate_pearson_k()


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global MODEL, ACTIVE_MODEL
        request = utils.receive_json(self.request)
        start = time.time()

        if request['action'] == 'set_model':
            model = request['model']
            if model in MODELS:
                ACTIVE_MODEL = request['model']
                MODEL = MODELS[ACTIVE_MODEL]()
            else:
                self.request.sendall(json.dumps({
                    'action': 'error',
                    'message': 'Incorrect model selected.'
                }).encode())
        elif request['action'] == 'get_model':
            self.request.sendall(json.dumps({
                'action': 'report',
                'model': ACTIVE_MODEL
            }).encode())
        elif request['action'] == 'build':
            success = True
            # noinspection PyBroadException
            try:
                MODEL.build(request['path'])
            except Exception:
                success = False
            self.request.sendall(json.dumps({
                'action': 'report',
                'success': success
            }).encode())
        elif request['action'] == 'query':
            self.request.sendall(
                MODEL.query(request['token'], request['query'], request['count']).encode()
            )
        elif request['action'] == 'update_query':
            self.request.sendall(
                MODEL.update_query(request['token'], request['document'], request['positive'],
                                   request['count']).encode()
            )
        else:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': 'Invalid action.'
            }).encode())

        print('Processed action "%s" in %.2f seconds' % (request['action'], time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('network')
    parser.add_argument('--model', default='Vector')
    args = parser.parse_args()

    ACTIVE_MODEL = args.model
    MODELS = {
        'Vector': Vector,
        'GeneralizedVector': GeneralizedVector
    }
    MODEL = MODELS[ACTIVE_MODEL]()
    NETWORK = json.load(open(args.network))

    server = socketserver.TCPServer((NETWORK['models']['host'], NETWORK['models']['port']), TCPHandler)
    server.serve_forever()
