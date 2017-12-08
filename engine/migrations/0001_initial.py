# Generated by Django 2.0 on 2017-12-08 03:29

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=140, unique=True)),
                ('title', models.CharField(max_length=140)),
                ('content', models.TextField(max_length=280)),
                ('path', models.FilePathField(unique=True)),
            ],
        ),
    ]
