import argparse
import os
from bisect import bisect_left

import django
import numpy as np

import engine.modules.ui as ui

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'info_retrieval.settings')
django.setup()


def e_measure(relevant, retrieved, beta):
    pre = precision(relevant, retrieved)
    rec = recall(relevant, retrieved)
    den = beta ** 2 * (pre + rec)
    return (((1 + beta ** 2) * pre * rec) / den) if den else 0


def f_measure(relevant, retrieved):
    pre = precision(relevant, retrieved)
    rec = recall(relevant, retrieved)
    den = pre + rec
    return ((2 * pre * rec) / den) if den else 0


def precision(relevant, retrieved):
    ra = np.sum(relevant[retrieved]) if retrieved.shape[0] > 0 else 0
    a = retrieved.shape[0]
    return ra / a if a else 0


def recall(relevant, retrieved):
    ra = np.sum(relevant[retrieved]) if retrieved.shape[0] > 0 else 0
    r = np.sum(relevant)
    return ra / r


def r_precision(relevant, retrieved):
    r = np.sum(relevant)
    return precision(relevant, retrieved[:r])


class Evaluator:
    def __init__(self, query_file, rel_file):
        self.queries = self._parse_queries(query_file)
        self.relevant = self._parse_relevant(rel_file)

    @staticmethod
    def _parse_queries(query_file):
        queries = []
        with open(query_file) as query_file:
            current = ''
            for line in query_file:
                if line[:2] == '.I' and current:
                    queries.append(current.strip())
                    current = ''
                if line[:2] in ('.I', '.W'):
                    continue
                else:
                    current += ' ' + line.strip()
        queries.append(current.strip())
        return queries

    def _parse_relevant(self, rel_file):
        relevant = [[] for _ in range(len(self.queries))]
        with open(rel_file) as rel_file:
            for line in rel_file:
                line = line.split()
                q = int(line[0]) - 1
                doc = line[1] + '.txt'
                relevant[q].append(doc)
        return relevant

    def evaluate(self, count):
        from engine.models import Document
        collection = [doc.filename for doc in Document.objects.all()]
        collection.sort()

        results = [0] * 6
        for i, q in enumerate(self.queries):
            response = ui.search(q, count)
            retrieved = response.get('results', [])
            retrieved = [doc['document'] for doc in retrieved]
            retrieved = [bisect_left(collection, doc) for doc in retrieved]

            rel = [False] * len(collection)
            for doc in self.relevant[i]:
                j = bisect_left(collection, doc)
                rel[j] = True

            retrieved = np.array(retrieved)
            relevant = np.array(rel)

            results[0] += precision(relevant, retrieved)
            results[1] += recall(relevant, retrieved)
            results[2] += f_measure(relevant, retrieved)
            results[3] += e_measure(relevant, retrieved, 0.5)
            results[4] += e_measure(relevant, retrieved, 1.5)
            results[5] += r_precision(relevant, retrieved)

        return np.array(results) / len(self.queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query_file')
    parser.add_argument('rel_file')
    args = parser.parse_args()

    evaluator = Evaluator(args.query_file, args.rel_file)
    print('precision - recall - f_measure - e_measure(beta=0.5) - e_measure(beta=1.5) - r_precision')
    for i in range(5, 110, 10):
        print('Count %d:\t%s' % (i, str(evaluator.evaluate(i))))
