import argparse
import json
import os
import pickle
import random
import socketserver
import time
from pathlib import Path

import numpy as np
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import engine.modules.utils as utils


class Summary:
    def __init__(self, path):
        self.path = path
        self.doc_names = [
            '.'.join(doc.name.split('.')[:-1])
            for doc in Path(self.path).iterdir() if doc.name.endswith('.txt')
        ]
        self.doc_names.sort()
        self.doc_count = len(self.doc_names)
        self.kmeans = self._build_kmeans()

    @staticmethod
    def _choose_k(X, start, end, step):
        scores = []
        for k in range(start, end + 1, step):
            kmeans = MiniBatchKMeans(n_clusters=k).fit(X)
            a = metrics.silhouette_score(X, kmeans.labels_)
            b = metrics.calinski_harabaz_score(X, kmeans.labels_)
            scores.append((a + b) / 2)
        return int(np.argmax(scores)) + 2

    def _build_kmeans(self):
        path = os.path.join(self.path, 'summary.bin')

        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
        else:
            index = self._load_index()
            freq, terms = utils.load_freq(index, self.doc_names)
            tfidf = TfidfTransformer().fit_transform(freq.transpose())

            # Dimensionality reduction using LSA (latent semantic analysis)
            svd = TruncatedSVD(n_components=min(len(terms), 100))
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            X = lsa.fit_transform(tfidf)

            # Deciding k
            k = self._choose_k(X, 2, min(200, self.doc_count - 1), (self.doc_count - 2) // 5)
            k = self._choose_k(X, max(2, k - 5), min(self.doc_count - 1, k + 5), 1)

            # Do the actual clustering
            kmeans = MiniBatchKMeans(n_clusters=k)
            kmeans.fit(X)

            with open(path, 'wb') as file:
                pickle.dump(kmeans, file)
            return kmeans

    def _load_index(self):
        while True:
            index = utils.send_json({
                'action': 'load',
                'path': self.path
            }, NETWORK['indices']['host'], NETWORK['indices']['port'], True)
            if index:
                break
            time.sleep(random.randint(10, 60))  # Give some time so indices module can make it
        return index

    def get(self):
        pass


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global SUMMARY
        request = utils.receive_json(self.request)
        start = time.time()

        if request['action'] == 'build':
            success = True
            # noinspection PyBroadException
            try:
                SUMMARY = Summary(request['path'])
            except Exception:
                success = False
            self.request.sendall(json.dumps({
                'action': 'report',
                'success': success
            }).encode())
        elif not SUMMARY:
            self.request.sendall(json.dumps({
                'action': 'error',
                'message': '"build" action required before any other action'
            }).encode())
        elif request['action'] == 'get':
            self.request.sendall(SUMMARY.get().encode())
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
    SUMMARY = None

    server = socketserver.TCPServer((NETWORK['summaries']['host'], NETWORK['summaries']['port']), TCPHandler)
    server.serve_forever()
