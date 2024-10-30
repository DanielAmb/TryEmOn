# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.index, name='index'),
# ]

from django.urls import path
from .views import ImageUploadView, index

urlpatterns = [
    path('', index, name='index'),
    path('upload/', ImageUploadView.as_view(), name='image-upload'),
]