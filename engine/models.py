from django.db import models


class Document(models.Model):
    file = models.FileField()


class Term(models.Model):
    name = models.CharField(max_length=140)
    k = models.FloatField()


class Occurrence(models.Model):
    term = models.ForeignKey(Term, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    weight = models.FloatField()
