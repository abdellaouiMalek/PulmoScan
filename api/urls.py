from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_image, name='predict_image'),
    path('predictions/', views.get_predictions, name='get_predictions'),
]
