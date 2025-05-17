import requests
import os
import json

def test_prediction_api(image_path, api_url="http://127.0.0.1:8000/api/predict/"):
    """Test the prediction API by uploading an image"""
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return False
    
    print(f"Testing API with image: {image_path}")
    
    # Prepare the files for upload
    files = {
        'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        # Make the API request
        print(f"Sending request to {api_url}...")
        response = requests.post(api_url, files=files)
        
        # Check the response
        if response.status_code == 200:
            print("SUCCESS: API request successful!")
            print(f"Response status code: {response.status_code}")
            
            # Parse the JSON response
            result = response.json()
            print("\nPrediction result:")
            print(f"  Prediction value: {result.get('prediction', 'N/A')}")
            print(f"  Is malignant: {result.get('is_malignant', 'N/A')}")
            print(f"  Image URL: {result.get('image_url', 'N/A')}")
            print(f"  Created at: {result.get('created_at', 'N/A')}")
            
            return True
        else:
            print(f"ERROR: API request failed with status code {response.status_code}")
            print(f"Response content: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception occurred during API request: {str(e)}")
        return False
    finally:
        # Close the file
        files['image'][1].close()

if __name__ == "__main__":
    # Test the API with the test image
    image_path = "test_image.jpg"
    success = test_prediction_api(image_path)
    
    if success:
        print("\nAPI test completed successfully!")
    else:
        print("\nAPI test failed!")
