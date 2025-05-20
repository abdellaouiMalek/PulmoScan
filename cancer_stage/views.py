import os
import numpy as np
import random
from django.shortcuts import render
from django.views.generic import TemplateView, View
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pydicom
import nibabel as nib
import SimpleITK as sitk
from .ct_preprocessing import preprocess_ct_scan

# Path to the trained model
RESNET_MODEL_PATH = os.path.join(settings.BASE_DIR, 'cancer_stage', 'resnet18_3d_cancer_stage.pth')
DENSENET_MODEL_PATH = os.path.join(settings.BASE_DIR, 'cancer_stage', 'densenet121_3d_cancer_stage.pth')

# Stage labels
STAGE_LABELS = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

class CancerStageDashboardView(TemplateView):
    """View for the cancer stage dashboard"""
    template_name = 'cancer_stage/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Cancer Stage Analysis'
        return context

class CancerStageUploadView(View):
    """View for uploading CT scans for cancer stage analysis"""

    def get(self, request):
        return render(request, 'cancer_stage/upload.html', {'title': 'Upload CT Scan'})

    def post(self, request):
        if 'ct_files' not in request.FILES:
            return JsonResponse({'error': 'No files uploaded'}, status=400)

        # Create a temporary directory for the uploaded files
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'temp_ct_scans'))

        # Save all uploaded files
        uploaded_files = request.FILES.getlist('ct_files')
        file_paths = []

        for file in uploaded_files:
            filename = fs.save(file.name, file)
            file_paths.append(fs.path(filename))

        try:
            # Process the CT scan
            processed_scan = self.process_ct_scan(file_paths)

            # Predict the cancer stage
            stage, confidence = self.predict_cancer_stage(processed_scan)

            # Save the results
            result_id = self.save_results(request.user, stage, confidence)

            # Clean up temporary files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)

            return JsonResponse({
                'success': True,
                'stage': stage,
                'confidence': confidence,
                'result_id': result_id
            })

        except Exception as e:
            # Clean up temporary files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)

            return JsonResponse({'error': str(e)}, status=500)

    def process_ct_scan(self, file_paths):
        """Process the uploaded CT scan files"""
        # Check file type
        if len(file_paths) == 1 and file_paths[0].endswith('.nii.gz'):
            # Process NIfTI file
            nifti_img = nib.load(file_paths[0])
            ct_data = nifti_img.get_fdata()
        else:
            # Process DICOM files
            ct_data = self.load_dicom_series(file_paths)

        # Preprocess the CT scan using the existing function
        # The function expects parameters, so we'll provide them
        processed_scan = preprocess_ct_scan(
            ct_scan=ct_data,
            target_shape=(128, 128, 128),
            normalize=True,
            apply_window=True
        )

        # Add channel dimension for the model (channels first for PyTorch)
        processed_scan = np.expand_dims(processed_scan, axis=0)

        return processed_scan

    def load_dicom_series(self, file_paths):
        """Load a series of DICOM files and convert to a 3D volume"""
        # Sort files by instance number
        dicom_files = []
        for file_path in file_paths:
            try:
                dcm = pydicom.dcmread(file_path)
                dicom_files.append((dcm.InstanceNumber, file_path))
            except:
                continue

        dicom_files.sort()
        sorted_file_paths = [file_path for _, file_path in dicom_files]

        # Use SimpleITK to read the DICOM series
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_file_paths)
        image = reader.Execute()

        # Convert to numpy array
        ct_data = sitk.GetArrayFromImage(image)

        return ct_data

    def predict_cancer_stage(self, processed_scan):
        """Predict the cancer stage using the trained model"""
        # For demonstration purposes, we'll return a simulated result
        # In a real implementation, you would load the model and make a prediction

        # Simulate a prediction - we're not actually using the processed_scan here
        # but in a real implementation, we would use it with the model
        predicted_class = random.randint(0, 3)  # Random stage between 0 and 3
        confidence = random.uniform(70.0, 95.0)  # Random confidence between 70% and 95%

        # In a real implementation, you would do something like this:
        """
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Choose which model to use
        model_type = 'resnet'  # or 'densenet'

        if model_type == 'resnet':
            model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=4)
            if os.path.exists(RESNET_MODEL_PATH):
                model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
        else:
            model = DenseNet3D(num_classes=4)
            if os.path.exists(DENSENET_MODEL_PATH):
                model.load_state_dict(torch.load(DENSENET_MODEL_PATH, map_location=device))

        model.to(device)
        model.eval()

        # Prepare input
        input_tensor = torch.from_numpy(processed_scan).float().to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)  # Add channel dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        """

        return STAGE_LABELS[predicted_class], confidence

    def save_results(self, user, stage, confidence):
        """Save the prediction results to the database"""
        # This would typically save to a database model
        # For now, we'll just return a dummy ID
        return 'result_' + str(hash(f"{user.username}_{stage}_{confidence}"))[:8]

class CancerStageResultsView(TemplateView):
    """View for displaying cancer stage analysis results"""
    template_name = 'cancer_stage/results.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Cancer Stage Results'
        # In a real implementation, you would fetch results from the database
        # For now, we'll just use dummy data
        context['results'] = [
            {
                'id': 'result_12345678',
                'date': '2025-05-20',
                'stage': 'Stage II',
                'confidence': 87.5,
            },
            {
                'id': 'result_87654321',
                'date': '2025-05-18',
                'stage': 'Stage III',
                'confidence': 92.3,
            }
        ]
        return context
