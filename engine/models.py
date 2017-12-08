from django.db import models


class Document(models.Model):
    filename = models.CharField(max_length=140, unique=True)
    title = models.CharField(max_length=140)
    content = models.TextField(max_length=280)
    path = models.FilePathField(unique=True)
