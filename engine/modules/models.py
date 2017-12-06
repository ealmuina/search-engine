from bisect import bisect_left
from functools import reduce

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Vector:
    def __init__(self, d):
        vectorizer = TfidfVectorizer()
        self.w = vectorizer.fit_transform(d).transpose()
        self.terms = vectorizer.get_feature_names()
        self.term_count = self.w.shape[0]
        self.doc_count = self.w.shape[1]

    def similarity(self, j, q):
        vectorizer = TfidfVectorizer(vocabulary=self.terms)
        q = vectorizer.fit_transform([q]).transpose()
        num = sum([self.w[i, j] * q[i] for i in range(self.term_count)])[0, 0]
        den = np.math.sqrt(reduce(lambda x, y: x + y ** 2, self.w[:, j], 0)[0, 0]) * np.math.sqrt(
            reduce(lambda x, y: x + y ** 2, q, 0)[0, 0])
        return num / den


class GeneralizedVector:
    def __init__(self, d):
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

    def similarity(self, j, q):
        vectorizer = TfidfVectorizer(vocabulary=self.terms)
        w = vectorizer.fit_transform([q]).transpose()
        q = reduce(
            lambda acum, i: acum + w[i, 0] * self.k[i],
            range(self.term_count),
            np.zeros(self.term_count)
        )
        d = reduce(lambda acum, i: acum + self.w[i, j] * self.k[i], range(self.term_count), np.zeros(self.term_count))

        return cosine_similarity(q.reshape(1, -1), d.reshape(1, -1))[0, 0]


def main():
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
        print(gv.similarity(j, q))
    print()
    for j in range(4):
        print(v.similarity(j, q))


if __name__ == '__main__':
    main()
