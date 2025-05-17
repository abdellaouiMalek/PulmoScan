from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from .models import PredictionResult
from .serializers import ScanImageSerializer, PredictionResponseSerializer
from . import lung_cancer_model
import logging

# Configure logging
logger = logging.getLogger(__name__)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])
def predict_image(request):
    """
    API endpoint for image prediction
    """
    logger.info("Received prediction request")

    if 'image' not in request.FILES:
        logger.error("No image provided in request")
        return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

    logger.info(f"Image received: {request.FILES['image'].name}, size: {request.FILES['image'].size} bytes")

    # Save the uploaded image
    image_serializer = ScanImageSerializer(data=request.data)
    if image_serializer.is_valid():
        logger.info("Image data is valid, saving...")
        scan_image = image_serializer.save()

        # Associate with user if authenticated
        if request.user.is_authenticated:
            scan_image.user = request.user
            scan_image.save()
            logger.info(f"Image associated with user: {request.user.username}")

        logger.info(f"Image saved at: {scan_image.image.path}")

        try:
            # Make prediction
            logger.info("Loading model and making prediction...")
            result = lung_cancer_model.predict(scan_image.image)
            logger.info(f"Prediction result: {result}")

            # Save prediction result
            prediction = PredictionResult.objects.create(
                scan_image=scan_image,
                prediction=result['prediction'],
                is_malignant=result['is_malignant']
            )
            logger.info(f"Prediction saved with ID: {prediction.id}")

            # Return the prediction result
            serializer = PredictionResponseSerializer(prediction, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            # Delete the image if prediction fails
            scan_image.delete()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        logger.error(f"Invalid image data: {image_serializer.errors}")

    return Response(image_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_predictions(request):
    """
    API endpoint to get all predictions
    """
    predictions = PredictionResult.objects.all().order_by('-created_at')
    serializer = PredictionResponseSerializer(predictions, many=True, context={'request': request})
    return Response(serializer.data)
