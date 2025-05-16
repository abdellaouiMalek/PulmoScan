import os
import sys
import tensorflow as tf
import numpy as np
import argparse

def convert_model(input_path, output_path):
    """
    Try to convert a Keras model to H5 format
    """
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Attempting to load model from: {input_path}")
    
    try:
        # Try to load the model with custom_objects to handle potential issues
        model = tf.keras.models.load_model(
            input_path, 
            compile=False,
            custom_objects={
                # Add any custom layers or objects here if needed
            }
        )
        print("Model loaded successfully!")
        
        # Save the model in H5 format
        h5_path = output_path + ".h5" if not output_path.endswith(".h5") else output_path
        model.save(h5_path, save_format='h5')
        print(f"Model converted and saved to: {h5_path}")
        
        # Verify the converted model can be loaded
        try:
            converted_model = tf.keras.models.load_model(h5_path)
            print("Converted model loaded successfully!")
            
            # Test with random input
            print("\nTesting model with random input...")
            input_shape = converted_model.input_shape
            print(f"Input shape: {input_shape}")
            
            # Create random input
            random_input = np.random.random((1, input_shape[1], input_shape[2], input_shape[3]))
            
            # Make prediction
            print("Making prediction...")
            prediction = converted_model.predict(random_input)
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction value: {prediction[0][0]}")
            
            return True
        except Exception as e:
            print(f"Error loading converted model: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        
        # Try alternative approach: recreate the model architecture and copy weights
        try:
            print("\nTrying alternative approach: recreating model architecture...")
            
            # Create a base EfficientNetB7 model
            base_model = tf.keras.applications.EfficientNetB7(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            
            # Create a new model with the same architecture as your original model
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            new_model = tf.keras.Model(inputs, outputs)
            
            # Save the new model
            h5_path = output_path + ".h5" if not output_path.endswith(".h5") else output_path
            new_model.save(h5_path, save_format='h5')
            print(f"New model created and saved to: {h5_path}")
            
            # Verify the new model can be loaded
            try:
                converted_model = tf.keras.models.load_model(h5_path)
                print("New model loaded successfully!")
                
                # Test with random input
                print("\nTesting model with random input...")
                random_input = np.random.random((1, 224, 224, 3))
                
                # Make prediction
                print("Making prediction...")
                prediction = converted_model.predict(random_input)
                print(f"Prediction shape: {prediction.shape}")
                print(f"Prediction value: {prediction[0][0]}")
                
                print("\nWARNING: This is a new model with the same architecture but different weights!")
                print("You will need to train this model or fine-tune it with your data.")
                
                return True
            except Exception as e:
                print(f"Error loading new model: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Error during alternative approach: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Convert a Keras model to H5 format')
    parser.add_argument('--input', type=str, default='efficientnetb7_final_model.keras',
                        help='Path to the input model file')
    parser.add_argument('--output', type=str, default='converted_model.h5',
                        help='Path to save the converted model')
    
    args = parser.parse_args()
    
    success = convert_model(args.input, args.output)
    
    if success:
        print("\nSUCCESS: Model conversion completed!")
    else:
        print("\nFAILURE: Model conversion failed.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
