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
    return render(request, 'engine/document_list.html', ui.search(query, count))
