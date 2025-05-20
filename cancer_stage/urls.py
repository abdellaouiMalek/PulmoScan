from django.urls import path
from . import views

urlpatterns = [
    path('', views.CancerStageDashboardView.as_view(), name='cancer_stage_dashboard'),
    path('upload/', views.CancerStageUploadView.as_view(), name='cancer_stage_upload'),
    path('results/', views.CancerStageResultsView.as_view(), name='cancer_stage_results'),
]
