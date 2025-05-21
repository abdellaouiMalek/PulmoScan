import preprocess.py
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.stats import gaussian_kde
import seaborn as sns
from skimage import morphology
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from Models.MedicalNet.models import resnet  # from MedicalNet/models/resnet.py
import torch.optim as optim
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def build_3d_yolov8(input_shape=(500, 500, 500, 1), num_classes=1):
    """
    Builds a 3D YOLOv8-inspired model for CT scan object detection.
    
    Args:
        input_shape: Shape of input volume (z, y, x, channels)
        num_classes: Number of output classes
        
    Returns:
        A TensorFlow Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial stem block
    x = layers.Conv3D(32, 3, strides=2, padding='same', kernel_regularizer=l2(5e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Backbone (modified CSPDarknet)
    def dark_conv(x, filters, kernel=3, strides=1):
        x = layers.Conv3D(filters, kernel, strides=strides, padding='same', 
                         kernel_regularizer=l2(5e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x
    
    def csp_block(x, filters, n_blocks):
        route = x
        route = dark_conv(route, filters // 2, kernel=1)
        
        x = dark_conv(x, filters // 2, kernel=1)
        for _ in range(n_blocks):
            x = dark_conv(x, filters // 2, kernel=1)
            x = dark_conv(x, filters // 2, kernel=3)
        
        x = layers.Concatenate()([route, x])
        x = dark_conv(x, filters, kernel=1)
        return x
    
    # Backbone layers
    x = dark_conv(x, 64, strides=2)
    x = csp_block(x, 64, n_blocks=1)
    
    x = dark_conv(x, 128, strides=2)
    x = csp_block(x, 128, n_blocks=2)
    
    x = dark_conv(x, 256, strides=2)
    x = csp_block(x, 256, n_blocks=2)
    
    x = dark_conv(x, 512, strides=2)
    x = csp_block(x, 512, n_blocks=1)
    
    # Head - predicting 7 values per detection:
    # (class_prob, z_center, y_center, x_center, z_size, y_size, x_size)
    outputs = layers.Conv3D(7, 1, activation='sigmoid', 
                           kernel_regularizer=l2(5e-4))(x)
    
    return models.Model(inputs, outputs, name='3d_yolov8')

# Custom loss function for YOLO-style detection
class Yolo3DLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_coord=5, lambda_size=5, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_size = lambda_size
        self.lambda_noobj = lambda_noobj
        
    def call(self, y_true, y_pred):
        # Split predictions
        pred_class = y_pred[..., 0:1]  # Class probability
        pred_boxes = y_pred[..., 1:]   # (z,y,x,dz,dy,dx)
        
        # Split ground truth
        true_class = y_true[..., 0:1]
        true_boxes = y_true[..., 1:]
        
        # Object mask (1 for objects, 0 for background)
        obj_mask = tf.cast(true_class > 0, tf.float32)
        
        # Class loss (binary crossentropy)
        class_loss = tf.keras.losses.binary_crossentropy(true_class, pred_class)
        class_loss = tf.reduce_mean(class_loss)
        
        # Coordinate loss (MSE for centers)
        coord_loss = tf.reduce_sum(tf.square(true_boxes[..., :3] - pred_boxes[..., :3]), axis=-1)
        coord_loss = self.lambda_coord * tf.reduce_mean(obj_mask * coord_loss)
        
        # Size loss (MSE for dimensions)
        size_loss = tf.reduce_sum(tf.square(tf.sqrt(true_boxes[..., 3:]) - 
                                 tf.sqrt(pred_boxes[..., 3:])), axis=-1)
        size_loss = self.lambda_size * tf.reduce_mean(obj_mask * size_loss)
        
        # No-object loss (penalize false positives)
        noobj_loss = self.lambda_noobj * tf.reduce_mean((1 - obj_mask) * 
                                         tf.keras.losses.binary_crossentropy(true_class, pred_class))
        
        return class_loss + coord_loss + size_loss + noobj_loss
    
# Compile the model
def compile_model(model, learning_rate=1e-4):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = Yolo3DLoss()
    model.compile(optimizer=optimizer, loss=loss)
    return model


# Initialize model
    yolo_v8 = build_3d_yolov8(input_shape=(500, 500, 500, 1))
    yolo_v8 = compile_model(yolo_v8)
    # yolo_v8.summary()
    
class ct_generator_yolo(Sequence):
    def __init__(self, ct_scans, annotations, batch_size, shuffle=True):
        """
        Initializes the data generator.
        
        ct_scans: A csv file containing all CT scans and its corresponding paths.
        annotations: Path to the annotations CSV file containing ct_scans and annotations.
        batch_size: Number of samples per batch.
        patch_size: Size of the extracted 3D patches (default is 64x64x64).
        shuffle: Whether to shuffle the data after each epoch.
        """

        self.ct_scans = load_paths(ct_scans)
        self.annotations = pd.read_csv(annotations)  # Load the candidate annotations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.annotations)) # Creating an array of indexes the same size of candidates
        self.target_shape = (500, 500, 500)  # Target shape for padding/cropping
        
        # print("Loaded CT scans :", len(self.ct_scans))
        # # print("Sample seriesuid from annotations:", self.annotations["seriesuid"].iloc[0])


        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """ Returns the number of batches per epoch """

        return int(np.floor(len(self.annotations) / self.batch_size))
        
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indexes[start_idx:end_idx]

        batch_ct_scans = []
        batch_labels = []
        ct_scan_cache = {}

        for i in batch_indices:
            scan = self.annotations.iloc[i]
            ct_scan_id = scan["seriesuid"]
            ct_scan_path = self.ct_scans[ct_scan_id]

            # Load CT scan (use cache if available)
            if ct_scan_path in ct_scan_cache:
                ct_scan = ct_scan_cache[ct_scan_path]
                resizedAnn = ct_scan_cache[ct_scan_path + "_ann"]
            else:
                ct_scan, numpyOrigin, numpySpacing = load_itk_image(ct_scan_path)

                # Resampling
                ct_scan, newOrigin, newSpacing, resizedAnn = resize_image_with_given_annotations(
                    ct_scan, numpyOrigin, numpySpacing, scan
                )

                # Update annotation coordinates
                bbox_z = resizedAnn["coordZ"].values[0]
                bbox_y = resizedAnn["coordY"].values[0]
                bbox_x = resizedAnn["coordX"].values[0]
                diameter_voxel_z = resizedAnn["diameter_voxel_z"].values[0]
                diameter_voxel_y = resizedAnn["diameter_voxel_y"].values[0]
                diameter_voxel_x = resizedAnn["diameter_voxel_x"].values[0]
                
                # print("Before padding:")
                # print("z: ", bbox_z, "y: ", bbox_y, "x: ", bbox_x)
                # print("diameter_voxel_z: ", diameter_voxel_z, "diameter_voxel_y: ", diameter_voxel_y, "diameter_voxel_x: ", diameter_voxel_x)

                # Clip HU values
                ct_scan = clip_CT_scan(ct_scan)

                # Segment lungs
                mask = isolate_lung(ct_scan)
                ct_scan = ct_scan * mask

                # Calculate padding amounts (no cropping)
                pad_dims = []
                for dim in range(3):
                    dim_size = ct_scan.shape[dim]
                    diff = self.target_shape[dim] - dim_size
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                    pad_dims.append((pad_before, pad_after))

                # Apply padding
                if any(p[0] != 0 or p[1] != 0 for p in pad_dims):
                    ct_scan = np.pad(ct_scan, pad_dims, mode='constant', constant_values=0)

                # Adjust bbox coordinates based on padding
                adjusted_bbox_z = bbox_z + pad_dims[0][0]
                adjusted_bbox_y = bbox_y + pad_dims[1][0]
                adjusted_bbox_x = bbox_x + pad_dims[2][0]

                # Verify coordinates are within bounds
                adjusted_bbox_z = np.clip(adjusted_bbox_z, 0, self.target_shape[0] - 1)
                adjusted_bbox_y = np.clip(adjusted_bbox_y, 0, self.target_shape[1] - 1)
                adjusted_bbox_x = np.clip(adjusted_bbox_x, 0, self.target_shape[2] - 1)
                
                # print("After padding:")
                # print("z: ", adjusted_bbox_z, "y: ", adjusted_bbox_y, "x: ", adjusted_bbox_x)
                # print("diameter_voxel_z: ", diameter_voxel_z, "diameter_voxel_y: ", diameter_voxel_y, "diameter_voxel_x: ", diameter_voxel_x)

                # Normalize
                ct_scan = Min_Max_scaling(ct_scan)

                # Cache results
                ct_scan_cache[ct_scan_path] = ct_scan
                ct_scan_cache[ct_scan_path + "_ann"] = resizedAnn

                # Normalize bbox & diameter relative to final shape
                bbox_z_norm = adjusted_bbox_z / self.target_shape[0]
                bbox_y_norm = adjusted_bbox_y / self.target_shape[1]
                bbox_x_norm = adjusted_bbox_x / self.target_shape[2]
                diameter_voxel_z_norm = diameter_voxel_z / self.target_shape[0]
                diameter_voxel_y_norm = diameter_voxel_y / self.target_shape[1]
                diameter_voxel_x_norm = diameter_voxel_x / self.target_shape[2]

            # Add channel dimension
            ct_scan = np.expand_dims(ct_scan, axis=-1)

            # Append to batch
            batch_ct_scans.append(ct_scan)
            batch_labels.append((
                0,  # Class ID
                bbox_z_norm,
                bbox_y_norm,
                bbox_x_norm,
                diameter_voxel_z_norm,
                diameter_voxel_y_norm,
                diameter_voxel_x_norm
            ))

        return np.stack(batch_ct_scans, axis=0), np.array(batch_labels)
    
    def on_epoch_end(self):
        """ Shuffle data after each epoch """
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            
train_generator_yolo = ct_generator_yolo(
        ct_scans = "Data/Luna/CT_scans.csv",
        annotations = "Data/Luna/input/train.csv",
        batch_size = 20,
    )

val_generator_yolo = ct_generator_yolo(
        ct_scans = "Data/Luna/CT_scans.csv",
        annotations = "Data/Luna/input/val.csv",
        batch_size = 20,
    )

test_generator_yolo = ct_generator_yolo(
        ct_scans = "Data/Luna/CT_scans.csv",
        annotations = "Data/Luna/input/test.csv",
        batch_size = 20,
    )

checkpoint_path = "Models/yolo_v8.h5"  # Path to save the checkpoint

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,  # Path to save the model
    save_weights_only=False,  # Save the entire model (architecture + weights + optimizer state)
    save_best_only=True,      # Save only the best model based on validation loss
    monitor='val_loss',       # Metric to monitor (e.g., validation loss)
    mode='min',              # Minimize the monitored metric
    verbose=1                # Print a message when saving the model
)

early_stop = EarlyStopping(
    monitor='val_loss',  # You can also monitor 'val_accuracy' or another metric
    patience=5,          # Number of epochs to wait for improvement before stopping
    restore_best_weights=True,  # Restore the model weights from the best epoch
    verbose=1            # To see when the early stopping is triggered
)

with tf.device('/GPU:0'):
    yolo_v8.fit(
        train_generator_yolo,
        validation_data=val_generator_yolo,
        epochs=30,
        verbose=1,
        callbacks=[checkpoint_callback, early_stop]  # Add the early stop callback
    )