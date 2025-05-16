import os
import sys
import argparse
import json
import shutil
from datetime import datetime
from predict_with_efficientnet import load_model, predict_image

def create_results_directory(base_dir="results"):
    """Create a directory to store results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f"pulmoscan_{timestamp}")
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    return result_dir

def copy_image_to_results(image_path, result_dir):
    """Copy the image to the results directory"""
    try:
        # Get the base filename
        base_name = os.path.basename(image_path)
        
        # Create the destination path
        dest_path = os.path.join(result_dir, base_name)
        
        # Copy the file
        shutil.copy2(image_path, dest_path)
        
        return dest_path
    except Exception as e:
        print(f"Error copying image: {str(e)}")
        return None

def save_result_to_file(result, result_dir):
    """Save the prediction result to a file"""
    try:
        # Create the result filename
        base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
        result_file = os.path.join(result_dir, f"{base_name}_result.json")
        
        # Save the result
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)
        
        return result_file
    except Exception as e:
        print(f"Error saving result: {str(e)}")
        return None

def create_report(result, result_dir):
    """Create a human-readable report"""
    try:
        # Create the report filename
        base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
        report_file = os.path.join(result_dir, f"{base_name}_report.txt")
        
        # Create the report content
        report_content = f"""
PulmoScan Analysis Report
========================

Image: {os.path.basename(result['image_path'])}
Date: {result['timestamp']}

Results:
--------
Prediction Value: {result['prediction']:.4f}
Classification: {result['result_text']}

Interpretation:
--------------
The image has been classified as {result['result_text'].lower()} with a confidence of {result['prediction']:.2%}.
{"This indicates a high probability of malignancy." if result['is_malignant'] else "This indicates a low probability of malignancy."}

Note: This is an automated analysis and should be reviewed by a medical professional.
"""
        
        # Save the report
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return report_file
    except Exception as e:
        print(f"Error creating report: {str(e)}")
        return None

def process_image(model, image_path, threshold=0.5):
    """Process a single image and save results"""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return False
    
    # Create results directory
    result_dir = create_results_directory()
    print(f"Results will be saved to: {result_dir}")
    
    # Copy image to results directory
    copied_image = copy_image_to_results(image_path, result_dir)
    if not copied_image:
        return False
    
    # Make prediction
    result = predict_image(model, image_path, threshold)
    if not result:
        return False
    
    # Save result to file
    result_file = save_result_to_file(result, result_dir)
    if not result_file:
        return False
    
    # Create report
    report_file = create_report(result, result_dir)
    if not report_file:
        return False
    
    # Print summary
    print("\nAnalysis completed successfully!")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {result['prediction']:.4f}")
    print(f"Classification: {result['result_text']}")
    print(f"\nResults saved to: {result_dir}")
    print(f"Report: {os.path.basename(report_file)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='PulmoScan CLI - Lung Cancer Detection')
    parser.add_argument('--model', type=str, default='efficientnetb7_final_model.keras',
                        help='Path to the EfficientNetB7 model file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the lung image to analyze')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return 1
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return 1
    
    # Process image
    success = process_image(model, args.image, args.threshold)
    
    return 0 if success else 1

if __name__ == "__main__":
    print("PulmoScan CLI - Lung Cancer Detection")
    print("=====================================")
    sys.exit(main())
