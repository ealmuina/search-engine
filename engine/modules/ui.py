from django.conf import settings

from engine.modules.utils import send_json


def build(path):
    dic = {
        'action': 'build',
        'path': path
    }
    send_json(dic, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'])
    send_json(dic, settings.NETWORK['recommendation']['host'], settings.NETWORK['recommendation']['port'])


def fit_suggestions(token, document):
    send_json({
        'action': 'fit',
        'token': token,
        'document': document
    }, settings.NETWORK['recommendation']['host'], settings.NETWORK['recommendation']['port'])


def get_model():
    return send_json({
        'action': 'get_model'
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'], True)


def get_suggestions(token):
    return send_json({
        'action': 'suggest',
        'token': token
    }, settings.NETWORK['recommendation']['host'], settings.NETWORK['recommendation']['port'], True)


def init(model):
    send_json({
        'action': 'init',
        'model': model
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'])


def search(query, count):
    return send_json({
        'action': 'query',
        'query': query,
        'count': count
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'], True)
