from django.db import models


class Document(models.Model):
    path = models.FilePathField(unique=True)


class Term(models.Model):
    name = models.CharField(unique=True, max_length=140)


class Correlation(models.Model):
    term1 = models.ForeignKey(Term, on_delete=models.CASCADE, related_name='correlation_set_term1')
    term2 = models.ForeignKey(Term, on_delete=models.CASCADE, related_name='correlation_set_term2')
    k = models.FloatField()

    class Meta:
        unique_together = ('term1', 'term2')


class Occurrence(models.Model):
    term = models.ForeignKey(Term, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    weight = models.FloatField()

    class Meta:
        unique_together = ('term', 'document')
