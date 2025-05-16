from rest_framework import serializers
from .models import ScanImage, PredictionResult

class ScanImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScanImage
        fields = ['id', 'image', 'uploaded_at']

class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = ['id', 'scan_image', 'prediction', 'is_malignant', 'created_at']
        
class PredictionResponseSerializer(serializers.ModelSerializer):
    """Serializer for prediction response with image URL included"""
    image_url = serializers.SerializerMethodField()
    result_text = serializers.SerializerMethodField()
    
    class Meta:
        model = PredictionResult
        fields = ['id', 'prediction', 'is_malignant', 'created_at', 'image_url', 'result_text']
    
    def get_image_url(self, obj):
        request = self.context.get('request')
        if obj.scan_image.image and hasattr(obj.scan_image.image, 'url'):
            return request.build_absolute_uri(obj.scan_image.image.url)
        return None
    
    def get_result_text(self, obj):
        return "Malin" if obj.is_malignant else "Bénin"
