import argparse
import json
import re
import socketserver
import time

import nltk
from nltk.corpus import stopwords, wordnet

from engine.modules.utils import receive_json

STOPS = set(stopwords.words('english'))
STEMMER = nltk.PorterStemmer()
LEMMATIZER = nltk.WordNetLemmatizer()


def _expand_query(words):
    query = []
    for word in words:
        syns = _get_synonyms(word)
        query.extend(syns)
    return query


def _get_synonyms(word, count=3):
    syns = {word}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if len(syns) > count:
                break
            syns.add(lemma.name())
    return syns


def _lem_stem(words):
    words = nltk.pos_tag(words)
    terms = []
    for word, tag in words:
        word = LEMMATIZER.lemmatize(word)
        if tag[0] not in ('N', 'J'):
            word = STEMMER.stem(word)
        terms.append(word)
    return terms


def _lexical_analysis(text):
    text = text.lower()
    words = re.findall('[a-z]+', text)
    return words


def _remove_stopwords(words):
    return [w for w in words if w not in STOPS]


def process(text, is_query):
    words = _lexical_analysis(text)
    words = _remove_stopwords(words)
    words = _lem_stem(words)

    if is_query:
        words = _expand_query(words)

    result = {
        'terms': words
    }
    return json.dumps(result)


class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        request = receive_json(self.request)
        start = time.time()

        if request['action'] == 'process':
            text = request['data']
            is_query = request.get('is_query', False)
            self.request.sendall(process(text, is_query).encode())
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

    server = socketserver.TCPServer((NETWORK['text']['host'], NETWORK['text']['port']), TCPHandler)
    server.serve_forever()
