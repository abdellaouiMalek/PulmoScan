from django.urls import path
from . import views

urlpatterns = [
    path('/classificationDashboard', views.DashboardView.as_view(), name='classificationDashboard'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('results/', views.ResultsView.as_view(), name='results'),
]
