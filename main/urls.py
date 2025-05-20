from django.urls import path
from . import views

urlpatterns = [
    # Home and general pages
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('services/', views.services, name='services'),
    path('doctors/', views.doctors, name='doctors'),
    path('contact/', views.contact, name='contact'),
    path('appointment/', views.appointment, name='appointment'),
    path('test/', views.test, name='test'),
    
    # User authentication
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # Dashboard and profile
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    
    # Nodule detection
    path('detection/', views.detection, name='detection'),
    path('process_nodules/', views.process_nodules, name='process_nodules'),
    
    # Visualization
    path('visualization/<str:filename>/', views.view_visualization, name='view_visualization'),
]
