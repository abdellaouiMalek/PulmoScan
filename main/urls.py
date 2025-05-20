from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.decorators import login_required
from dashboard.views import DashboardView, UploadView, ResultsView


urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('services/', views.services, name='services'),
    path('doctors/', views.doctors, name='doctors'),
    path('contact/', views.contact, name='contact'),
    path('appointment/', views.appointment, name='appointment'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('test/', views.test, name='test'),
    path('medical/', include('medical_app.urls')),

    path('dashboardClassification/', login_required(DashboardView.as_view()), name='dashboardClassification'),
    path('upload/', login_required(UploadView.as_view()), name='upload'),
    path('results/', login_required(ResultsView.as_view()), name='results'),
    
]
