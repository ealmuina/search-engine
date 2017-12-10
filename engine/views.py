import pathlib
import time
from bisect import bisect_left

import numpy as np
from django.shortcuts import render, HttpResponse

import engine.evaluation as evaluation
import engine.modules.ui as ui
from engine.models import Document


def build(request):
    path = request.GET.get('path')
    path_docs = set(doc.name for doc in pathlib.Path(path).iterdir())
    db_docs = set(doc.filename for doc in Document.objects.all())

    if path_docs != db_docs:
        bulk = []
        for doc in pathlib.Path(path).iterdir():
            if doc.name == 'index.json':
                continue
            with open(str(doc)) as file:
                title = file.readline(140)
                content = file.read(280)
            bulk.append(Document(
                path=str(doc),
                filename=doc.name,
                title=title,
                content=content
            ))
        Document.objects.all().delete()
        Document.objects.bulk_create(bulk)

    ui.build(path)
    return HttpResponse()


def evaluate(request):
    return render(request, 'engine/evaluation.html', {'documents': Document.objects.all()})


def get_evaluations(request):
    collection = [doc.filename for doc in Document.objects.all()]
    collection.sort()

    relevant = request.GET.getlist('relevant[]')
    count = int(request.GET.get('count'))
    query = request.GET.get('query')
    beta = float(request.GET.get('beta'))

    response = ui.search(query, count)
    retrieved = response.get('results', [])
    retrieved = [doc['document'] for doc in retrieved]

    retrieved = [bisect_left(collection, doc) for doc in retrieved]
    rel = [False] * len(collection)
    for doc in relevant:
        j = bisect_left(collection, doc)
        rel[j] = True

    retrieved = np.array(retrieved)
    relevant = np.array(rel)

    return render(request, 'engine/evaluation_report.html', {
        'precision': evaluation.precision(relevant, retrieved),
        'recall': evaluation.recall(relevant, retrieved),
        'f_measure': evaluation.f_measure(relevant, retrieved),
        'e_measure': evaluation.e_measure(relevant, retrieved, beta),
        'r_precision': evaluation.r_precision(relevant, retrieved)
    })


def get_model(request):
    response = ui.get_model()
    return HttpResponse(response['model'])


def index(request):
    return render(request, 'engine/index.html')


def init(request):
    model = request.GET.get('model')
    ui.init(model)
    return HttpResponse()


def search(request):
    start = time.time()
    query = request.GET.get('q')
    count = int(request.GET.get('count'))
    response = ui.search(query, count)
    results = []
    if response['success']:
        results = [(Document.objects.get(filename=doc['document']), doc['match']) for doc in response['results']]
    return render(request, 'engine/document_list.html', {
        'query': query,
        'documents': results,
        'time': round(time.time() - start, 2)
    })
