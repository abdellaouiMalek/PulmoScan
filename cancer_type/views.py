import os
import numpy as np
import random
from django.shortcuts import render
from django.views.generic import TemplateView, View
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import PIL.Image as Image

# Path to the trained model
RESNET_MODEL_PATH = os.path.join(settings.BASE_DIR, 'cancer_type', 'resnet18_cancer_type.pth')

# Cancer type labels
CANCER_TYPE_LABELS = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Small Cell Carcinoma', 'Normal']

class CancerTypeDashboardView(TemplateView):
    """View for the cancer type dashboard"""
    template_name = 'cancer_type/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Cancer Type Analysis'
        return context

class CancerTypeUploadView(View):
    """View for uploading SVS images for cancer type analysis"""

    def get(self, request):
        return render(request, 'cancer_type/upload.html', {'title': 'Upload SVS Image'})

    def post(self, request):
        if 'svs_file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        # Create a temporary directory for the uploaded file
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp_svs_images'))

        # Save the uploaded file
        uploaded_file = request.FILES['svs_file']
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            # Process the SVS image
            processed_image = self.process_svs_image(file_path)

            # Predict the cancer type
            cancer_type, confidence = self.predict_cancer_type(processed_image)

            # Save the results - use a username if available, otherwise use 'anonymous'
            username = request.user.username if request.user.is_authenticated else 'anonymous'
            result_id = self.save_results(username, cancer_type, confidence)

            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            return JsonResponse({
                'success': True,
                'cancer_type': cancer_type,
                'confidence': confidence,
                'result_id': result_id
            })

        except Exception as e:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Log the error for debugging
            import traceback
            print(f"Error in cancer_type upload: {str(e)}")
            print(traceback.format_exc())

            return JsonResponse({'error': str(e)}, status=500)

    def process_svs_image(self, file_path):
        """Process the uploaded SVS image"""
        # In a real implementation, you would use openslide to process the SVS image
        # For demonstration purposes, we'll just return a dummy processed image

        # Try to open the image with PIL to check if it's a valid image
        try:
            img = Image.open(file_path)
            img_array = np.array(img)
            return img_array
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    def predict_cancer_type(self, processed_image):
        """Predict the cancer type using the trained model"""
        # For demonstration purposes, we'll return a simulated result
        # In a real implementation, you would load the model and make a prediction

        # Simulate a prediction - we're not actually using the processed_image here
        # but in a real implementation, we would use it with the model
        predicted_class = random.randint(0, 3)  # Random type between 0 and 3
        confidence = random.uniform(70.0, 95.0)  # Random confidence between 70% and 95%

        return CANCER_TYPE_LABELS[predicted_class], confidence

    def save_results(self, username, cancer_type, confidence):
        """Save the prediction results to the database"""
        # This would typically save to a database model
        # For now, we'll just return a dummy ID
        return 'result_' + str(hash(f"{username}_{cancer_type}_{confidence}"))[:8]

class CancerTypeResultsView(TemplateView):
    """View for displaying cancer type analysis results"""
    template_name = 'cancer_type/results.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Cancer Type Results'
        # In a real implementation, you would fetch results from the database
        # For now, we'll just use dummy data
        context['results'] = [
            {
                'id': 'result_12345678',
                'date': '2025-05-20',
                'cancer_type': 'Adenocarcinoma',
                'confidence': 87.5,
                'image_url': '/static/img/sample_svs.jpg'
            },
            {
                'id': 'result_87654321',
                'date': '2025-05-18',
                'cancer_type': 'Squamous Cell Carcinoma',
                'confidence': 92.3,
                'image_url': '/static/img/sample_svs.jpg'
            }
        ]
        return context
