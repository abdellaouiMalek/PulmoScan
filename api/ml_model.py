import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO
from django.conf import settings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Set TensorFlow log level
tf.get_logger().setLevel('ERROR')

class LungCancerModel:
    """Class to handle the model for lung cancer prediction"""

    def __init__(self):
        self.model = None
        # Use the demo model instead of the original model
        self.model_path = "demo_model"
        self.target_size = (224, 224)  # Standard input size
        self.threshold = 0.5  # Threshold for binary classification

    def load(self):
        """Load the model from disk"""
        if self.model is None:
            try:
                logger.info(f"Loading model from {self.model_path}")
                # Check if model file exists
                if not os.path.exists(self.model_path):
                    error_msg = f"Model file not found at {self.model_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                self.model = load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}", exc_info=True)
                raise

    def preprocess_image(self, img):
        """Preprocess the image for model input"""
        # Resize image
        img = img.resize(self.target_size)

        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]

        return img_array

    def predict(self, img_file):
        """Make a prediction on the given image file"""
        logger.info(f"Starting prediction for image: {img_file}")

        # Load model if not already loaded
        if self.model is None:
            logger.info("Model not loaded yet, loading now...")
            self.load()

        try:
            # Open and preprocess the image
            logger.info("Opening and preprocessing image...")
            img = Image.open(img_file).convert('RGB')
            processed_img = self.preprocess_image(img)

            # Make prediction
            logger.info("Running prediction...")
            prediction = self.model.predict(processed_img, verbose=0)[0][0]
            logger.info(f"Raw prediction value: {prediction}")

            # Determine if malignant based on threshold
            is_malignant = bool(prediction >= self.threshold)
            logger.info(f"Classification result: {'Malignant' if is_malignant else 'Benign'} (threshold: {self.threshold})")

            return {
                'prediction': float(prediction),
                'is_malignant': is_malignant
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise

# Create a singleton instance
model = LungCancerModel()
