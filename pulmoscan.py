#!/usr/bin/env python
import os
import sys
import argparse
import subprocess

def print_header():
    """Print the PulmoScan header"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██╗   ██╗██╗     ███╗   ███╗ ██████╗ ███████╗ ██████╗ █████╗ ███╗   ██╗  ║
║   ██╔══██╗██║   ██║██║     ████╗ ████║██╔═══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║  ║
║   ██████╔╝██║   ██║██║     ██╔████╔██║██║   ██║███████╗██║     ███████║██╔██╗ ██║  ║
║   ██╔═══╝ ██║   ██║██║     ██║╚██╔╝██║██║   ██║╚════██║██║     ██╔══██║██║╚██╗██║  ║
║   ██║     ╚██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝███████║╚██████╗██║  ██║██║ ╚████║  ║
║   ╚═╝      ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝  ║
║                                                               ║
║                Lung Cancer Detection System                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def print_menu():
    """Print the main menu"""
    print("\nPulmoScan - Main Menu")
    print("====================")
    print("1. Launch Web Interface")
    print("2. Launch GUI Interface")
    print("3. Command Line Interface")
    print("4. Batch Processing")
    print("5. Convert Model")
    print("6. Exit")
    print()

def launch_web_interface():
    """Launch the Django web interface"""
    print("\nLaunching Web Interface...")
    try:
        subprocess.run([sys.executable, "manage.py", "runserver"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching web interface: {str(e)}")
    except KeyboardInterrupt:
        print("\nWeb interface stopped.")

def launch_gui_interface():
    """Launch the GUI interface"""
    print("\nLaunching GUI Interface...")
    try:
        subprocess.run([sys.executable, "pulmoscan_gui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching GUI interface: {str(e)}")
    except KeyboardInterrupt:
        print("\nGUI interface stopped.")

def launch_cli():
    """Launch the command line interface"""
    print("\nCommand Line Interface")
    print("=====================")

    # Get model path
    model_path = input("Enter model path (default: models/efficientnetb7_final_model.keras): ")
    if not model_path:
        model_path = "models/efficientnetb7_final_model.keras"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return

    # Get image path
    image_path = input("Enter image path: ")
    if not image_path:
        print("Error: No image path provided")
        return

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return

    # Get threshold
    threshold = input("Enter threshold (default: 0.5): ")
    if not threshold:
        threshold = "0.5"

    # Launch CLI
    try:
        subprocess.run([
            sys.executable, "pulmoscan_cli.py",
            "--model", model_path,
            "--image", image_path,
            "--threshold", threshold
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in CLI: {str(e)}")
    except KeyboardInterrupt:
        print("\nCLI stopped.")

def launch_batch_processing():
    """Launch batch processing"""
    print("\nBatch Processing")
    print("===============")

    # Get model path
    model_path = input("Enter model path (default: models/efficientnetb7_final_model.keras): ")
    if not model_path:
        model_path = "models/efficientnetb7_final_model.keras"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return

    # Get input directory
    input_dir = input("Enter input directory containing images: ")
    if not input_dir:
        print("Error: No input directory provided")
        return

    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} not found")
        return

    # Get output file
    output_file = input("Enter output file (default: batch_results.json): ")
    if not output_file:
        output_file = "batch_results.json"

    # Get threshold
    threshold = input("Enter threshold (default: 0.5): ")
    if not threshold:
        threshold = "0.5"

    # Launch batch processing
    try:
        subprocess.run([
            sys.executable, "batch_predict.py",
            "--model", model_path,
            "--input", input_dir,
            "--output", output_file,
            "--threshold", threshold
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in batch processing: {str(e)}")
    except KeyboardInterrupt:
        print("\nBatch processing stopped.")

def convert_model():
    """Convert model to H5 format"""
    print("\nModel Conversion")
    print("===============")

    # Get input model path
    input_path = input("Enter input model path (default: models/efficientnetb7_final_model.keras): ")
    if not input_path:
        input_path = "models/efficientnetb7_final_model.keras"

    # Check if model exists
    if not os.path.exists(input_path):
        print(f"Error: Model file {input_path} not found")
        return

    # Get output model path
    output_path = input("Enter output model path (default: converted_model.h5): ")
    if not output_path:
        output_path = "converted_model.h5"

    # Launch model conversion
    try:
        subprocess.run([
            sys.executable, "convert_model_h5.py",
            "--input", input_path,
            "--output", output_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in model conversion: {str(e)}")
    except KeyboardInterrupt:
        print("\nModel conversion stopped.")

def main():
    parser = argparse.ArgumentParser(description='PulmoScan - Lung Cancer Detection System')
    parser.add_argument('--web', action='store_true', help='Launch web interface')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--cli', action='store_true', help='Launch command line interface')
    parser.add_argument('--batch', action='store_true', help='Launch batch processing')
    parser.add_argument('--convert', action='store_true', help='Convert model to H5 format')

    args = parser.parse_args()

    print_header()

    # Check if any argument is provided
    if args.web:
        launch_web_interface()
        return
    elif args.gui:
        launch_gui_interface()
        return
    elif args.cli:
        launch_cli()
        return
    elif args.batch:
        launch_batch_processing()
        return
    elif args.convert:
        convert_model()
        return

    # If no argument is provided, show menu
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            launch_web_interface()
        elif choice == '2':
            launch_gui_interface()
        elif choice == '3':
            launch_cli()
        elif choice == '4':
            launch_batch_processing()
        elif choice == '5':
            convert_model()
        elif choice == '6':
            print("\nExiting PulmoScan. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPulmoScan terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)
