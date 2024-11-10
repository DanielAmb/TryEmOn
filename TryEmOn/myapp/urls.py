from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
]


from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),  # Include the URLs from `myapp`
]
