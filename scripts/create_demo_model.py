import tensorflow as tf
import numpy as np
import os

def create_simple_model():
    """
    Create a simple CNN model for demonstration purposes
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Simple model created successfully!")
    model.summary()
    
    # Save the model
    model_path = "demo_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    model_path = create_simple_model()
    
    # Test the model with random input
    print("\nTesting model with random input...")
    random_input = np.random.random((1, 224, 224, 3))
    
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(random_input)
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction value: {prediction[0][0]}")
    print("\nSUCCESS: Model created and tested successfully!")
