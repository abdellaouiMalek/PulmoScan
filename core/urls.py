"""
URL configuration for PulmoScan project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.decorators import login_required
from dashboard.views import DashboardView, UploadView, ResultsView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('accounts/', include('accounts.urls')),
    path('medical/', include('medical_app.urls')),

    path('dashboardClassification/', login_required(DashboardView.as_view()), name='dashboard'),
    path('upload/', login_required(UploadView.as_view()), name='upload'),
    path('results/', login_required(ResultsView.as_view()), name='results'),
    path("admin/", admin.site.urls),
    path("", include("main.urls")),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
