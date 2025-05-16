from django.urls import path
from . import views

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('results/', views.ResultsView.as_view(), name='results'),
]
