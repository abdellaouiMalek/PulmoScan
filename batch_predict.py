import os
import sys
import argparse
import json
import csv
from datetime import datetime
from predict_with_efficientnet import load_model, predict_image

def process_directory(model, directory, threshold=0.5, output_dir=None):
    """Process all images in a directory"""
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found")
        return None
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {directory}")
        return None
    
    print(f"Found {len(image_files)} image files in {directory}")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {image_path}")
        result = predict_image(model, image_path, threshold)
        
        if result:
            results.append(result)
            
            # Save individual result if output directory is specified
            if output_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_file = os.path.join(output_dir, f"{base_name}_result.json")
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=4)
                print(f"Result saved to {output_file}")
    
    return results

def save_batch_results(results, output_file):
    """Save batch results to a file"""
    if not results:
        print("No results to save.")
        return
    
    # Determine file extension
    _, ext = os.path.splitext(output_file)
    
    if ext.lower() == '.csv':
        # Save as CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    print(f"\nBatch results saved to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch predict lung cancer using EfficientNetB7 model')
    parser.add_argument('--model', type=str, default='efficientnetb7_final_model.keras',
                        help='Path to the EfficientNetB7 model file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the directory containing images to predict')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification')
    parser.add_argument('--output', type=str, default='batch_results.json',
                        help='Path to save the batch prediction results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save individual prediction results')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return 1
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return 1
    
    # Process directory
    results = process_directory(model, args.input, args.threshold, args.output_dir)
    if results is None:
        return 1
    
    # Save batch results
    save_batch_results(results, args.output)
    
    # Print summary
    malignant_count = sum(1 for r in results if r['is_malignant'])
    benign_count = len(results) - malignant_count
    
    print("\nBatch Processing Summary:")
    print(f"Total images processed: {len(results)}")
    print(f"Malignant: {malignant_count} ({malignant_count/len(results)*100:.1f}%)")
    print(f"Benign: {benign_count} ({benign_count/len(results)*100:.1f}%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
