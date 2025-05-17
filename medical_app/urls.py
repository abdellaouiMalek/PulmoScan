from django.urls import path
from . import views

urlpatterns = [
    path('', views.MedicalHomeView.as_view(), name='medical_home'),
    path('service-details/', views.MedicalServiceView.as_view(), name='medical_service'),
    path('starter-page/', views.MedicalStarterView.as_view(), name='medical_starter'),
]
