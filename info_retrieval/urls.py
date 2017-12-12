"""info_retrieval URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import engine.views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', engine.views.index),
    path('index/', engine.views.index, name='index'),

    path('set_model/', engine.views.set_model, name='set_model'),
    path('get_model/', engine.views.get_model, name='get_model'),
    path('build/', engine.views.build, name='build'),
    path('search/', engine.views.search, name='search'),

    path('evaluate/', engine.views.evaluate, name='evaluate'),
    path('get_evaluations/', engine.views.get_evaluations, name='get_evaluations'),

    path('visit/<str:document>', engine.views.visit, name='visit'),
    path('suggest/', engine.views.suggest, name='suggest'),
]
