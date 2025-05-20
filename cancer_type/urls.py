from django.urls import path
from . import views

urlpatterns = [
    path('', views.CancerTypeDashboardView.as_view(), name='cancer_type_dashboard'),
    path('upload/', views.CancerTypeUploadView.as_view(), name='cancer_type_upload'),
    path('results/', views.CancerTypeResultsView.as_view(), name='cancer_type_results'),
]
