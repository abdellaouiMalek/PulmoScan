from django.shortcuts import render
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

class MedicalHomeView(TemplateView):
    """View for the medical template home page"""
    template_name = 'medical/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'PulmoScan - Medical Dashboard'
        return context

class MedicalServiceView(TemplateView):
    """View for the medical service details page"""
    template_name = 'medical/service-details.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'PulmoScan - Service Details'
        return context

class MedicalStarterView(TemplateView):
    """View for the medical starter page"""
    template_name = 'medical/starter-page.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'PulmoScan - Starter Page'
        return context
