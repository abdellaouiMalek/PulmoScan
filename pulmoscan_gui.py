import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from predict_with_efficientnet import load_model, predict_image

class PulmoScanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PulmoScan - Lung Cancer Detection")
        self.root.geometry("900x700")
        self.root.minsize(900, 700)

        # Model variables
        self.model = None
        self.model_path = "models/efficientnetb7_final_model.keras"
        self.threshold = 0.5

        # Image variables
        self.image_path = None
        self.image = None
        self.photo = None

        # Result variables
        self.result = None

        # Create UI
        self.create_ui()

        # Load model in background
        self.status_var.set("Loading model...")
        threading.Thread(target=self.load_model_thread).start()

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="PulmoScan - Lung Cancer Detection", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create tabs
        tab_control = ttk.Notebook(main_frame)

        # Analysis tab
        analysis_tab = ttk.Frame(tab_control)
        tab_control.add(analysis_tab, text="Analysis")

        # Settings tab
        settings_tab = ttk.Frame(tab_control)
        tab_control.add(settings_tab, text="Settings")

        # History tab
        history_tab = ttk.Frame(tab_control)
        tab_control.add(history_tab, text="History")

        tab_control.pack(expand=1, fill=tk.BOTH)

        # Analysis tab content
        self.create_analysis_tab(analysis_tab)

        # Settings tab content
        self.create_settings_tab(settings_tab)

        # History tab content
        self.create_history_tab(history_tab)

    def create_analysis_tab(self, parent):
        # Top frame for buttons
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=10)

        # Load image button
        load_btn = ttk.Button(top_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)

        # Analyze button
        self.analyze_btn = ttk.Button(top_frame, text="Analyze Image", command=self.analyze_image, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        # Save result button
        self.save_btn = ttk.Button(top_frame, text="Save Result", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Middle frame for image and result
        middle_frame = ttk.Frame(parent)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Image frame
        image_frame = ttk.LabelFrame(middle_frame, text="Image")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Image canvas
        self.image_canvas = tk.Canvas(image_frame, bg="white")
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Result frame
        result_frame = ttk.LabelFrame(middle_frame, text="Result")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Result text
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, width=40, height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)

    def create_settings_tab(self, parent):
        # Settings frame
        settings_frame = ttk.Frame(parent, padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)

        # Model path
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="Model Path:").pack(side=tk.LEFT, padx=5)

        self.model_path_var = tk.StringVar(value=self.model_path)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        model_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        browse_btn = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        browse_btn.pack(side=tk.LEFT, padx=5)

        # Threshold
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=5)

        ttk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)

        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                   variable=self.threshold_var, length=300)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        threshold_label = ttk.Label(threshold_frame, textvariable=self.threshold_var)
        threshold_label.pack(side=tk.LEFT, padx=5)

        # Apply button
        apply_btn = ttk.Button(settings_frame, text="Apply Settings", command=self.apply_settings)
        apply_btn.pack(pady=10)

    def create_history_tab(self, parent):
        # History frame
        history_frame = ttk.Frame(parent, padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True)

        # History list
        ttk.Label(history_frame, text="Analysis History:").pack(anchor=tk.W, pady=5)

        # Create treeview
        columns = ("date", "image", "result", "prediction")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings")

        # Define headings
        self.history_tree.heading("date", text="Date")
        self.history_tree.heading("image", text="Image")
        self.history_tree.heading("result", text="Result")
        self.history_tree.heading("prediction", text="Prediction")

        # Define columns
        self.history_tree.column("date", width=150)
        self.history_tree.column("image", width=200)
        self.history_tree.column("result", width=100)
        self.history_tree.column("prediction", width=100)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)

        # Pack
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons frame
        buttons_frame = ttk.Frame(history_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        # View button
        view_btn = ttk.Button(buttons_frame, text="View Selected", command=self.view_history_item)
        view_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_btn = ttk.Button(buttons_frame, text="Clear History", command=self.clear_history)
        clear_btn.pack(side=tk.LEFT, padx=5)

    def load_model_thread(self):
        try:
            self.status_var.set(f"Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)

            if self.model:
                self.status_var.set("Model loaded successfully!")
            else:
                self.status_var.set("Failed to load model.")
                messagebox.showerror("Error", f"Failed to load model from {self.model_path}")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Error loading model: {str(e)}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )

        if file_path:
            try:
                self.image_path = file_path
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")

                # Load and display image
                self.display_image(file_path)

                # Enable analyze button
                self.analyze_btn.config(state=tk.NORMAL)

                # Clear result
                self.result = None
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.config(state=tk.DISABLED)
                self.save_btn.config(state=tk.DISABLED)

            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
                messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def display_image(self, image_path):
        # Clear canvas
        self.image_canvas.delete("all")

        # Load image
        img = Image.open(image_path)

        # Resize image to fit canvas
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 1:  # Canvas not yet drawn
            canvas_width = 400
            canvas_height = 400

        # Calculate new size while maintaining aspect ratio
        img_width, img_height = img.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to PhotoImage
        self.image = img
        self.photo = ImageTk.PhotoImage(img)

        # Display image
        self.image_canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)

    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded.")
            return

        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return

        try:
            self.status_var.set("Analyzing image...")

            # Disable buttons during analysis
            self.analyze_btn.config(state=tk.DISABLED)

            # Run analysis in a separate thread
            threading.Thread(target=self.analyze_thread).start()

        except Exception as e:
            self.status_var.set(f"Error analyzing image: {str(e)}")
            messagebox.showerror("Error", f"Error analyzing image: {str(e)}")
            self.analyze_btn.config(state=tk.NORMAL)

    def analyze_thread(self):
        try:
            # Make prediction
            self.result = predict_image(self.model, self.image_path, self.threshold)

            # Update UI
            self.root.after(0, self.update_result)

        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error analyzing image: {str(e)}"))

    def update_result(self):
        if not self.result:
            self.status_var.set("Analysis failed.")
            return

        # Update result text
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "Analysis Result\n", "heading")
        self.result_text.insert(tk.END, "==============\n\n")

        self.result_text.insert(tk.END, f"Image: {os.path.basename(self.image_path)}\n\n")

        self.result_text.insert(tk.END, f"Prediction: {self.result['prediction']:.4f}\n")
        self.result_text.insert(tk.END, f"Classification: {self.result['result_text']}\n\n")

        self.result_text.insert(tk.END, "Interpretation:\n")
        self.result_text.insert(tk.END, f"The image has been classified as {self.result['result_text'].lower()} ")
        self.result_text.insert(tk.END, f"with a confidence of {self.result['prediction']:.2%}.\n")

        if self.result['is_malignant']:
            self.result_text.insert(tk.END, "This indicates a high probability of malignancy.\n")
        else:
            self.result_text.insert(tk.END, "This indicates a low probability of malignancy.\n")

        self.result_text.insert(tk.END, "\nNote: This is an automated analysis and should be reviewed by a medical professional.")

        self.result_text.config(state=tk.DISABLED)

        # Enable save button
        self.save_btn.config(state=tk.NORMAL)

        # Re-enable analyze button
        self.analyze_btn.config(state=tk.NORMAL)

        # Update status
        self.status_var.set("Analysis completed.")

        # Add to history
        self.add_to_history()

    def add_to_history(self):
        # Add to treeview
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image = os.path.basename(self.image_path)
        result = self.result['result_text']
        prediction = f"{self.result['prediction']:.4f}"

        self.history_tree.insert("", 0, values=(date, image, result, prediction))

    def save_result(self):
        if not self.result:
            messagebox.showerror("Error", "No result to save.")
            return

        # Ask for directory
        directory = filedialog.askdirectory(title="Select Directory to Save Result")

        if not directory:
            return

        try:
            # Create result filename
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            result_file = os.path.join(directory, f"{base_name}_result.json")
            report_file = os.path.join(directory, f"{base_name}_report.txt")

            # Save result as JSON
            with open(result_file, 'w') as f:
                json.dump(self.result, f, indent=4)

            # Save report as text
            report_content = self.result_text.get(1.0, tk.END)
            with open(report_file, 'w') as f:
                f.write(report_content)

            self.status_var.set(f"Result saved to {directory}")
            messagebox.showinfo("Success", f"Result saved to:\n{result_file}\n{report_file}")

        except Exception as e:
            self.status_var.set(f"Error saving result: {str(e)}")
            messagebox.showerror("Error", f"Error saving result: {str(e)}")

    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("Keras Model", "*.keras *.h5")]
        )

        if file_path:
            self.model_path_var.set(file_path)

    def apply_settings(self):
        # Get new settings
        new_model_path = self.model_path_var.get()
        new_threshold = self.threshold_var.get()

        # Check if model path changed
        if new_model_path != self.model_path:
            self.model_path = new_model_path

            # Load new model
            threading.Thread(target=self.load_model_thread).start()

        # Update threshold
        self.threshold = new_threshold

        self.status_var.set("Settings applied.")

    def view_history_item(self):
        selected_item = self.history_tree.selection()

        if not selected_item:
            messagebox.showinfo("Info", "No item selected.")
            return

        # Get values
        values = self.history_tree.item(selected_item, "values")

        # Show in message box
        messagebox.showinfo("History Item", f"Date: {values[0]}\nImage: {values[1]}\nResult: {values[2]}\nPrediction: {values[3]}")

    def clear_history(self):
        # Clear treeview
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        self.status_var.set("History cleared.")

    def show_error(self, message):
        self.status_var.set(message)
        messagebox.showerror("Error", message)
        self.analyze_btn.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = PulmoScanGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
