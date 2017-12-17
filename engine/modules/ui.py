from django.conf import settings

from engine.modules.utils import send_json


def build(path):
    dic = {
        'action': 'build',
        'path': path
    }
    success = {
        send_json(dic, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'], True)['success'],
        send_json(dic, settings.NETWORK['recommendation']['host'], settings.NETWORK['recommendation']['port'], True)[
            'success'],
        send_json(dic, settings.NETWORK['summaries']['host'], settings.NETWORK['summaries']['port'], True)['success']
    }
    return False not in success


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


def get_summary():
    return send_json({
        'action': 'get'
    }, settings.NETWORK['summaries']['host'], settings.NETWORK['summaries']['port'], True)


def set_model(model):
    send_json({
        'action': 'set_model',
        'model': model
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'])


def search(token, query, count):
    return send_json({
        'action': 'query',
        'token': token,
        'query': query,
        'count': count
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'], True)


def update_search(token, document, positive, count):
    return send_json({
        'action': 'update_query',
        'token': token,
        'document': document,
        'positive': positive,
        'count': count
    }, settings.NETWORK['models']['host'], settings.NETWORK['models']['port'], True)
