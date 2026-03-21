# Import necessary libraries
import kagglehub
import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator

from tf_keras import layers, models
from tf_keras.preprocessing.image import *
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tf_keras.optimizers import Adam
from tf_keras.applications import MobileNetV2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
from sklearn.metrics import classification_report, confusion_matrix

def create_cnn_model(input_shape=(100, 100, 3), num_classes=None):
    """
    Create a custom CNN architecture for fruit classification
    Architecture:
    - 4 Convolutional blocks with increasing filters
    - Batch Normalization after each conv layer
    - MaxPooling for dimensionality reduction
    - Dropout for regularization
    - Global Average Pooling instead of Flatten
    - Dense layers with dropout for classification
    """

    if num_classes is None:
        num_classes = len(class_names)

    model = models.Sequential([
        # 1st convolutionalblock
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape, name='conv1_1'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2, name='pool1'),
        layers.Dropout(0.25, name='dropout1'),

        # second convolutionalblock
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2, name='pool2'),
        layers.Dropout(0.25, name='dropout2'),

        # 3rd convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2, name='pool3'),
        layers.Dropout(0.25, name='dropout3'),

        # fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2, name='pool4'),
        layers.Dropout(0.25, name='dropout4'),

        # Classification head
        layers.GlobalAveragePooling2D(name='gap'),
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5, name='dropout5'),
        layers.Dense(256, activation='relu', name='dense2'),
        layers.BatchNormalization(),
        layers.Dropout(0.5, name='dropout6'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='Fruits360_CNN')

    return model

# Download latest version
path = kagglehub.dataset_download("moltean/fruits")
path = os.path.join(path, 'fruits-360_100x100', 'fruits-360')

print("Path to dataset files:", path)
# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Check your dataset structure
print("\n" + "=" * 60)
print("CHECKING DATASET STRUCTURE")
print("=" * 60)

# List all files in the input directory
print("\nContents of dataset path:")
for item in os.listdir(path):
    print(f"- {item}")

    # If it's a directory, show its contents
    item_path = os.path.join(path, item)
    if os.path.isdir(item_path):
        for subitem in os.listdir(item_path)[:5]:  # Show first 5 items
            print(f"{subitem}")
# finding the correct path
possible_paths = [
    path,
]
# Find which path contains Training and Test folders
dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        print(f"\nFound dataset at: {p}")
        if os.path.exists(os.path.join(p, 'Training')) and os.path.exists(os.path.join(p, 'Test')):
            dataset_path = p
            print(f"   ✓ Contains Training and Test folders")
            break
        else:
            # Check subdirectories
            for item in os.listdir(p):
                subpath = os.path.join(p, item)
                if os.path.isdir(subpath):
                    if os.path.exists(os.path.join(subpath, 'Training')):
                        dataset_path = subpath
                        print(f"   ✓ Found Training in: {subpath}")
                        break

# Set training and test directories
train_dir = os.path.join(dataset_path, 'Training')
test_dir = os.path.join(dataset_path, 'Test')

print(f"Training directory: {train_dir}")
print(f"Test directory: {test_dir}")

print("="*60)
print(" EXPLORING DATASET")
print("="*60)

#Get all class names
classes = sorted(os.listdir(train_dir))
n_classes = len(classes)
print(f"\nDataset Statistics:")
print(f"Total number of classes: {n_classes}")
print(f"First 10 classes: {classes[:10]}")
print(f"Last 10 classes: {classes[-10:]}")

#Display sample images from different classes
print("\nDisplaying sample images from dataset...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()
for i in range(10):
    #Selecting random class and image
    class_name = random.choice(classes)
    class_path = os.path.join(train_dir, class_name)

    # Fix: doar fisiere imagine, nu foldere
    images = [f for f in os.listdir(class_path)
              if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        continue

    img_name = random.choice(images)
    img_path = os.path.join(class_path, img_name)

    #Loading and display image
    img = plt.imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"{class_name}\n{img.shape}", fontsize=10, fontweight='bold')
    axes[i].axis('off')
plt.suptitle('Sample Images from Fruits-360 Dataset', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("="*60)
print("DATA PREPROCESSING AND AUGMENTATION")
print("="*60)

# Image parameters
IMG_SIZE = 100  # Fruits-360 images are 100x100
BATCH_SIZE = 32
EPOCHS = 30

print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")

# Data augmentation for training (helps prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0,1]
    rotation_range=20,            # Randomly rotate images
    width_shift_range=0.2,        # Randomly shift images horizontally
    height_shift_range=0.2,       # Randomly shift images vertically
    shear_range=0.2,              # Shear transformations
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Random horizontal flips
    fill_mode='nearest',          # Fill mode for new pixels
)

# Only rescaling for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
print("\n Data generators created successfully!")
print(f"training samples: {train_generator.samples}")
print(f"test samples: {test_generator.samples}")
print(f"number of classes: {train_generator.num_classes}")

print("=" * 60)
print(" BUILDING CUSTOM CNN MODEL")
print("=" * 60)

# Creating the model
print("\n building CNN model...")
class_names = list(train_generator.class_indices.keys())
model = create_cnn_model(num_classes=n_classes)

# displaying model architecture
print("\n Model summary:")
model.summary()

# compile the model
print("\n compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("model compiled successfully!")

print("=" * 60)
print("Configuring training callbacks")
print("=" * 60)

# create callbacks for intelligent training
callbacks = [

    # considered with a early stoping: stoping training when validation loss stops improving
    EarlyStopping(
        monitor='loss',
        patience=7,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),

    # redecuing the learing rate . decreasing LR when plateau is detected
    ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    ),

    # modelcheckpoint-saving the best model during training
    ModelCheckpoint(
        filepath='best_fruit_model.h5',  # fix si path-ul pentru Windows
        monitor='accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("\n callbacks configured:")
print("1.early Stopping - patience: 7 epochs")
print("2.Reduce LR on Plateau - factor: 0.2, patience: 3")
print("3.model checkpoint - saving best model based on validation accuracy")

print("\nBest model will be saved to: /kaggle/working/best_fruit_model.h5")

print("="*60)
print("TRAINING THE MODEL")
print("="*60)

print(f"Training Configuration:")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Training samples: {train_generator.samples}")
print("="*60)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")

# training history plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# accuracy plotting
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# loss plotting
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTraining Summary:")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")