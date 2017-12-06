from django.shortcuts import render

import engine.modules.ui as ui


def index(request):
    return render(request, 'engine/index.html')


def search(request):
    query = request.GET.get('q')
    count = 10
    return render(request, 'engine/document_list.html', ui.search(query, count))
