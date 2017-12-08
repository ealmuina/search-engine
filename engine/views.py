from django.shortcuts import render, HttpResponse

import engine.modules.ui as ui


def build(request):
    path = request.GET.get('path')
    ui.build(path)
    return HttpResponse()


def index(request):
    return render(request, 'engine/index.html')


def search(request):
    query = request.GET.get('q')
    count = 10
    response = ui.search(query, count)
    results = []
    if response['success']:
        results = [(doc['document'], doc['match']) for doc in response['results']]
    return render(request, 'engine/document_list.html', {'documents': results})
