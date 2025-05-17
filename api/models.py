from django.db import models
from django.contrib.auth.models import User
import uuid
import os

def get_upload_path(instance, filename):
    """Generate a unique path for uploaded images"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads', filename)

class ScanImage(models.Model):
    """Model for storing uploaded scan images"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='scan_images')
    image = models.ImageField(upload_to=get_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        user_info = f" - {self.user.username}" if self.user else ""
        return f"Scan {self.id}{user_info} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

class PredictionResult(models.Model):
    """Model for storing prediction results"""
    scan_image = models.ForeignKey(ScanImage, on_delete=models.CASCADE, related_name='predictions')
    prediction = models.FloatField()  # Probability score
    is_malignant = models.BooleanField()  # True if malignant, False if benign
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        result = "Malin" if self.is_malignant else "Bénin"
        return f"Prédiction: {result} ({self.prediction:.2f}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
