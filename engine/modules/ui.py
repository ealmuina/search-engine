from django.conf import settings

from engine.modules.utils import send_json


def build(path):
    send_json({
        'action': 'build',
        'path': path
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'])


def get_model():
    return send_json({
        'action': 'get_model'
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'], True)


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
