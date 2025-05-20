from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_ct_nodules, name='detect_ct_nodules'),
    path('test/', views.test_endpoint, name='test_endpoint'),
]
