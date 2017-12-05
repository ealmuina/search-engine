from functools import reduce
from math import sqrt

from sklearn.feature_extraction.text import TfidfVectorizer


class Vector:
    def __init__(self, d):
        vectorizer = TfidfVectorizer()
        self.w = vectorizer.fit_transform(d).transpose()
        self.terms = vectorizer.get_feature_names()
        self.count = self.w.shape[0]

    def similarity(self, j, q):
        vectorizer = TfidfVectorizer(vocabulary=self.terms)
        q = vectorizer.fit_transform([q]).transpose()

        num = sum([self.w[i, j] * q[i] for i in range(self.count)])[0, 0]
        den = sqrt(reduce(lambda x, y: x + y ** 2, self.w[:, j], 0)[0, 0]) * sqrt(
            reduce(lambda x, y: x + y ** 2, q, 0)[0, 0])
        return num / den


def main():
    v = Vector([
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
    ])
    q = 'this is the second document'
    print(v.similarity(1, q))


if __name__ == '__main__':
    main()
