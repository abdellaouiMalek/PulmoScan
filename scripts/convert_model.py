import tensorflow as tf
import os
import sys

def convert_model(input_path, output_path):
    """
    Convert a Keras model to a format compatible with the current TensorFlow version
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
        
        # Save the model in the current TensorFlow format
        model.save(output_path, save_format='tf')
        print(f"Model converted and saved to: {output_path}")
        
        # Verify the converted model can be loaded
        try:
            converted_model = tf.keras.models.load_model(output_path)
            print("Converted model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading converted model: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "converted_model"
    else:
        input_path = "efficientnetb7_final_model.keras"
        output_path = "converted_model"
    
    success = convert_model(input_path, output_path)
    
    if success:
        print("\nSUCCESS: Model converted successfully!")
    else:
        print("\nFAILURE: Model conversion failed.")
