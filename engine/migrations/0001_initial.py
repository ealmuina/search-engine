# Generated by Django 2.0 on 2017-12-12 22:34

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Directory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.FilePathField(unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=140)),
                ('title', models.CharField(max_length=140)),
                ('content', models.TextField(max_length=280)),
                ('visits', models.PositiveIntegerField(default=0)),
                ('extension', models.CharField(max_length=10)),
                ('directory', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='engine.Directory')),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='document',
            unique_together={('directory', 'filename')},
        ),
    ]
