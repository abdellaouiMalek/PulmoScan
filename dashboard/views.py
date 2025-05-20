from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Avg
from django.utils import timezone
from datetime import timedelta
from api.models import PredictionResult

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboardClassification.html'  # Using the template from frontend/templates

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get user's predictions if authenticated
        if self.request.user.is_authenticated:
            predictions = PredictionResult.objects.filter(
                scan_image__user=self.request.user
            ).order_by('-created_at')
        else:
            predictions = PredictionResult.objects.all().order_by('-created_at')

        # Get recent predictions
        context['recent_predictions'] = predictions[:10]

        # Get statistics
        total_predictions = predictions.count()
        malignant_count = predictions.filter(is_malignant=True).count()
        benign_count = total_predictions - malignant_count

        # Calculate average confidence
        avg_confidence = "N/A"
        if total_predictions > 0:
            avg_confidence_value = predictions.aggregate(avg=Avg('prediction'))['avg']
            if avg_confidence_value is not None:
                avg_confidence = f"{avg_confidence_value:.2%}"

        # Get time-based statistics
        now = timezone.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        weekly_stats = predictions.filter(created_at__gte=week_ago).count()
        monthly_stats = predictions.filter(created_at__gte=month_ago).count()

        context['stats'] = {
            'total': total_predictions,
            'malignant': malignant_count,
            'benign': benign_count,
            'malignant_percent': (malignant_count / total_predictions * 100) if total_predictions > 0 else 0,
            'benign_percent': (benign_count / total_predictions * 100) if total_predictions > 0 else 0,
            'avg_confidence': avg_confidence,
            'weekly': weekly_stats,
            'monthly': monthly_stats
        }

        return context

class UploadView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/upload.html'

class ResultsView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/results.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get user's predictions if authenticated
        if self.request.user.is_authenticated:
            predictions = PredictionResult.objects.filter(
                scan_image__user=self.request.user
            ).order_by('-created_at')
        else:
            predictions = PredictionResult.objects.all().order_by('-created_at')

        context['predictions'] = predictions

        # Add filtering capabilities
        context['filter_options'] = {
            'date_ranges': [
                {'value': 'all', 'label': 'Toutes les dates'},
                {'value': 'today', 'label': 'Aujourd\'hui'},
                {'value': 'week', 'label': 'Cette semaine'},
                {'value': 'month', 'label': 'Ce mois'},
            ],
            'result_types': [
                {'value': 'all', 'label': 'Tous les résultats'},
                {'value': 'malignant', 'label': 'Malins uniquement'},
                {'value': 'benign', 'label': 'Bénins uniquement'},
            ]
        }

        return context
