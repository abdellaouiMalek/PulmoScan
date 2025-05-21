from django.urls import path
from . import views

urlpatterns = [
    path('', views.CancerDetectionDashboardView.as_view(), name='cancer_detection_dashboard'),
]
