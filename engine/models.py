import os

from django.db import models


class Directory(models.Model):
    path = models.FilePathField(unique=True)

    def __str__(self):
        return self.path


class Document(models.Model):
    directory = models.ForeignKey(Directory, on_delete=models.CASCADE)
    filename = models.CharField(max_length=140)
    title = models.CharField(max_length=140)
    content = models.TextField(max_length=280)
    visits = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ('directory', 'filename')

    @property
    def path(self):
        return os.path.join(self.directory.path, self.filename)

    def __str__(self):
        return self.filename
