import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import json
import time
from datetime import datetime

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for EfficientNetB7"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize image
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def load_model(model_path):
    """Load the EfficientNetB7 model"""
    try:
        print(f"Loading model from {model_path}...")
        
        # Try to load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        # Print model summary
        model.summary()
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image_path, threshold=0.5):
    """Make a prediction on an image using the model"""
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None
        
        print(f"Making prediction on image: {image_path}")
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Determine if malignant based on threshold
        is_malignant = bool(prediction >= threshold)
        
        result = {
            'image_path': image_path,
            'prediction': float(prediction),
            'is_malignant': is_malignant,
            'result_text': "Malin" if is_malignant else "Bénin",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def save_result(result, output_file=None):
    """Save the prediction result to a file"""
    if result is None:
        print("No result to save.")
        return
    
    # Print result to console
    print("\nPrediction Result:")
    print(f"Image: {result['image_path']}")
    print(f"Prediction value: {result['prediction']:.4f}")
    print(f"Result: {result['result_text']} (is_malignant: {result['is_malignant']})")
    print(f"Timestamp: {result['timestamp']}")
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\nResult saved to {output_file}")
        except Exception as e:
            print(f"Error saving result to file: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict lung cancer using EfficientNetB7 model')
    parser.add_argument('--model', type=str, default='efficientnetb7_final_model.keras',
                        help='Path to the EfficientNetB7 model file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to predict')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the prediction result (JSON format)')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
        return 1
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return 1
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return 1
    
    # Make prediction
    result = predict_image(model, args.image, args.threshold)
    if result is None:
        return 1
    
    # Save result
    save_result(result, args.output)
    
    return 0

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    sys.exit(main())
