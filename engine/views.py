import pathlib

from django.shortcuts import render, HttpResponse

from engine.models import Document
import engine.modules.ui as ui


def build(request):
    path = request.GET.get('path')
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


def index(request):
    return render(request, 'engine/index.html')


def search(request):
    query = request.GET.get('q')
    count = 10
    response = ui.search(query, count)
    results = []
    if response['success']:
        results = [(Document.objects.get(filename=doc['document']), doc['match']) for doc in response['results']]
    return render(request, 'engine/document_list.html', {'query': query, 'documents': results})
