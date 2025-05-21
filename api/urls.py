from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_image, name='predict_image'),
    path('predictions/', views.get_predictions, name='get_predictions'),
    path('analyze/', views.predict_image, name='analyze_image'),  # Add alias for /api/analyze/
]
