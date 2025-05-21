from django.views.generic import TemplateView, View
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin

class CancerDetectionDashboardView(LoginRequiredMixin, TemplateView):
    """View for the cancer detection dashboard"""
    template_name = 'cancer_detection/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Cancer Detection'
        return context
