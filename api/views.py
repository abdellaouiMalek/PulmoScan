import os
import pandas as pd
import logging
import tempfile
import sys
import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

# Add a unique timestamp to verify code is being reloaded
RELOAD_TIMESTAMP = time.time()

# Get the logger
logger = logging.getLogger('django')  # Use Django's logger

@csrf_exempt
def detect_ct_nodules(request):
    """
    API endpoint that logs information about uploaded files.
    """
    print("==== Starting detect_ct_nodules function ====")
    print(f"Code reload timestamp: {RELOAD_TIMESTAMP}")
    print(f"Current file path: {__file__}")
    
    if request.method != 'POST':
        print("Error: Only POST method is allowed")
        return JsonResponse({'error': 'Only POST method is allowed'}, status=400)
    
    # Print directly to console for immediate visibility
    print("==== File Upload Request Received ====")
    
    # Log using Django's logger
    logger.info("Received file upload request")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    print("dghgndg")
    
    # Return a simple response to verify the function is being called
    return JsonResponse({
        'message': 'API endpoint reached successfully',
        'timestamp': RELOAD_TIMESTAMP,
        'file_path': __file__
    })
    
    # The rest of the code is commented out to simplify debugging
    # If you see this response, it means your changes are being loaded

@csrf_exempt
def test_endpoint(request):
    """
    Simple test endpoint to verify API is working
    """
    print("==== Test endpoint called ====")
    return JsonResponse({
        'message': 'Test endpoint reached successfully',
        'timestamp': time.time(),
        'method': request.method
    })
