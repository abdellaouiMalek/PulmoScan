import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Chemin vers le modèle
model_path = 'efficientnetb7_final_model.keras'

def test_model_loading():
    """Test if the model can be loaded correctly"""
    print(f"Checking if model file exists at {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    print(f"Model file exists. Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Print model summary
        print("\nModel summary:")
        model.summary()
        
        # Test with random input
        print("\nTesting model with random input...")
        input_shape = model.input_shape
        print(f"Input shape: {input_shape}")
        
        # Create random input
        random_input = np.random.random((1, 224, 224, 3))
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(random_input)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction value: {prediction[0][0]}")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to load model: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing model loading...")
    success = test_model_loading()
    if success:
        print("\nSUCCESS: Model loaded and tested successfully!")
    else:
        print("\nFAILURE: Model could not be loaded or tested.")
