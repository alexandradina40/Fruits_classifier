# Import necessary libraries
import kagglehub
import tensorflow as tf

# Load the best model saved during training
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def predict_and_display(image_path, model, class_names, img_size=100):
    """
    Predict class for a single image and return results
    """  # Load and preprocess image
    # ✅ Corect - folosești funcțiile importate direct

    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_classes = [class_names[i] for i in top_3_idx]
    top_3_confidences = [predictions[0][i] for i in top_3_idx]
    return {
        'image': img,
        'predicted_class': class_names[predicted_idx],
        'confidence': confidence,
        'top_3_classes': top_3_classes,
        'top_3_confidences': top_3_confidences
    }

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
            print("   ✓ Contains Training and Test folders")
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
print("\nDataset Statistics:")
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

print("Training Configuration:")
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

print("="*60)
print("EVALUATING MODEL ON TEST SET")
print("="*60)

print("Loading best model from checkpoint...")
best_model = load_model('best_fruit_model.h5')
print("Best model loaded successfully!")
#Evaluate on test set
print("\nRunning evaluation on test data...")
test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)
print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Test Accuracy Percentage: {test_accuracy*100:.2f}%")
#Checking if target achieved
if test_accuracy > 0.80:
    print("\nSUCCESS!")
    print(f"model achieved {test_accuracy*100:.2f}% accuracy")
    print("target of >80% has been MET!")
else:
    print(f"\nModel achieved {test_accuracy*100:.2f}% accuracy")
    print(f"target of >80% is {(80 - test_accuracy*100):.2f}% away")

print("="*60)
print("DETAILED PERFORMANCE ANALYSIS")
print("="*60)

# Generate predictions
print("Generating predictions on test set...")
test_generator.reset()
predictions = best_model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Get the actual number of classes
actual_n_classes = len(np.unique(true_classes))
print(f"\nNumber of classes in test set: {actual_n_classes}")

# method (1)Show classification report for ALL classes (can be very long)
print("\nFull Classification Report (all classes):")
print("-" * 80)
report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,  # Use all class names
    zero_division=0
)
print(report)

#method 2: Show only the first 15 classes (safer approach)
print("\n" + "="*60)
print("Classification Report (First 15 Classes Only):")
print("="*60)
#Get indices for first 15 classes
first_15_indices = list(range(min(15, len(class_names))))
first_15_class_names = [class_names[i] for i in first_15_indices]
#Filter predictions for only these classes
mask = np.isin(true_classes, first_15_indices)
filtered_true = true_classes[mask]
filtered_pred = predicted_classes[mask]
print("\nShowing results for first 15 classes only:")
print(f"Total samples in these classes: {len(filtered_true)}")
if len(filtered_true) > 0:
    report_filtered = classification_report(
        filtered_true,
        filtered_pred,
        labels=first_15_indices,
        target_names=first_15_class_names,
        zero_division=0
    )
    print(report_filtered)
else:
    print("No samples found for first 15 classes in test set!")

#method (3) Show summary statistics instead
print("\n" + "="*60)
print("PERFORMANCE SUMMARY STATISTICS")
print("="*60)
from collections import Counter
#calculate overall accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"Overall Test Accuracy: {accuracy*100:.2f}%")
#calculate per-class accuracy for first 15 classes
print(f"\npre class accuracy (First {min(15, len(class_names))} Classes):")
print("-" * 60)
print(f"{'Class Name':<30} {'Samples':<10} {'Correct':<10} {'Accuracy':<10}")
print("-" * 60)
for i in range(min(15, len(class_names))):
    class_name = class_names[i]
    class_mask = (true_classes == i)
    class_samples = np.sum(class_mask)
    if class_samples > 0:
        class_correct = np.sum(predicted_classes[class_mask] == i)
        class_acc = class_correct / class_samples
        print(f"{class_name:<30} {class_samples:<10} {class_correct:<10} {class_acc*100:<10.2f}%")
    else:
        print(f"{class_name:<30} {'0':<10} {'0':<10} {'N/A':<10}")
#Find best and worst performing classes
print("\nTop 5 Best Performing Classes:")
class_accuracies = []
for i in range(len(class_names)):
    class_mask = (true_classes == i)
    if np.sum(class_mask) > 5:  # Only consider classes with enough samples
        class_acc = np.mean(predicted_classes[class_mask] == i)
        class_accuracies.append((class_names[i], class_acc, np.sum(class_mask)))
#Sort by accuracy
class_accuracies.sort(key=lambda x: x[1], reverse=True)
for i, (name, acc, samples) in enumerate(class_accuracies[:5]):
    print(f"{i+1}. {name:<30}Accuracy: {acc*100:.2f}% (Samples: {samples})")
print("\nBottom 5 Worst Performing Classes:")
for i, (name, acc, samples) in enumerate(class_accuracies[-5:]):
    print(f"   {i+1}. {name:<30} Accuracy: {acc*100:.2f}% (Samples: {samples})")
#Confusion Matrix (First 15 classes)
print("\ngenerating Confusion Matrix (First 15 classes)...")
#Filter for first 15 classes
first_15_mask = np.isin(true_classes, first_15_indices)
cm_true = true_classes[first_15_mask]
cm_pred = predicted_classes[first_15_mask]
if len(cm_true) > 0:
    cm = confusion_matrix(cm_true, cm_pred, labels=first_15_indices)
    plt.figure(figsize=(20, 16))
    #Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=first_15_class_names,
        yticklabels=first_15_class_names,
        annot_kws={'size': 10}
    )
    plt.title('confusion Matrix (First 15 Classes)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('predicted Label', fontsize=14)
    plt.ylabel('true Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
    #calculate confusion matrix statistics
    print("\nConfusion Matrix Statistics:")
    #Perclass accuracy from confusion matrix
    row_sums = cm.sum(axis=1)
    per_class_acc = cm.diagonal() / (row_sums + 1e-10)  # Add small epsilon to avoid division by zero
    for i, class_name in enumerate(first_15_class_names):
        if row_sums[i] > 0:
            print(f"   {class_name:<30} Accuracy: {per_class_acc[i]*100:.2f}%")
else:
    print("no samples from first 15 classes found in test set!")
    print("showing confusion matrix for all classes might be too large.")
    print("\nalternative: View confusion matrix for top 15 most frequent classes")

    #Find top 15 most frequent classes in test set
    class_counts = Counter(true_classes)
    top_15_classes = [cls for cls, count in class_counts.most_common(15)]
    top_15_names = [class_names[cls] for cls in top_15_classes]
    print("\nTop 15 most frequent classes in test set:")
    for i, cls in enumerate(top_15_classes):
        print(f"   {i+1}. {class_names[cls]}: {class_counts[cls]} samples")
    #Create confusion matrix for top 15 classes
    top_15_mask = np.isin(true_classes, top_15_classes)
    cm_true_top = true_classes[top_15_mask]
    cm_pred_top = predicted_classes[top_15_mask]
    #Remap class indices to 0-14 for confusion matrix
    remap_dict = {old: new for new, old in enumerate(top_15_classes)}
    cm_true_remapped = np.array([remap_dict[x] for x in cm_true_top])
    cm_pred_remapped = np.array([remap_dict[x] for x in cm_pred_top])
    cm_top = confusion_matrix(cm_true_remapped, cm_pred_remapped, labels=range(15))
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm_top,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=top_15_names,
        yticklabels=top_15_names,
        annot_kws={'size': 10}
    )
    plt.title('confusion Matrix (Top 15 Most Frequent Classes)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

print("=" * 60)
print("TESTING ON SAMPLE IMAGES")
print("=" * 60)

# Test on random images from test set
print("Testing model on random test images...")

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.ravel()

correct_predictions = 0
total_tested = 10

for i in range(total_tested):
    # Get random test image
    class_name = random.choice(classes)
    class_path = os.path.join(test_dir, class_name)
    img_name = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, img_name)

    # Get prediction
    result = predict_and_display(img_path, best_model, class_names)

    # Display image
    axes[i].imshow(result['image'])

    # Color code based on correctness
    if result['predicted_class'] == class_name:
        color = 'green'
        correct_predictions += 1
        status = 'CORRECT'
    else:
        color = 'red'
        status = 'WRONG'

    # Create title with prediction info
    title = f"True: {class_name[:15]}\n"
    title += f"Pred: {result['predicted_class'][:15]}\n"
    title += f"Conf: {result['confidence']:.2f}\n"
    title += f"{status}"

    axes[i].set_title(title, color=color, fontsize=9, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('odel predoction on test mages (Green=Correct, Red=Incorrect)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\nSample Test Results:")
print(f"Correct predictions: {correct_predictions}/{total_tested}")
print(f"Accuracy on samples: {correct_predictions / total_tested * 100:.1f}%")

print("=" * 60)
print("TESTING ON 200 SAMPLE IMAGES")
print("=" * 60)

# FIX: Get valid test classes first
print("Finding valid test classes...")
# Get all classes from train directory
all_classes = sorted(os.listdir(train_dir))
print(f"Total classes in train: {len(all_classes)}")
# Get all classes that actually exist in test directory
valid_test_classes = []
for class_name in all_classes:
    test_class_path = os.path.join(test_dir, class_name)
    if os.path.exists(test_class_path) and os.path.isdir(test_class_path):
        valid_test_classes.append(class_name)
print(f"Total classes found in test: {len(valid_test_classes)}")
print(f"Sample valid classes: {valid_test_classes[:10]}")
if len(valid_test_classes) == 0:
    print("ERROR: No valid test classes found!")
    print("Let's see what's actually in the test directory:")
    print(os.listdir(test_dir)[:20])  # Show first 20 items in test dir
    # Stop execution
    raise SystemExit("Cannot proceed without test classes")
# Use only valid test classes
classes = valid_test_classes
# Test on random images from test set
print(f"\nTesting model on random test images from {len(classes)} valid classes...")
# Create figure for displaying images (smaller grid for display)
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.ravel()
correct_predictions = 0
total_tested = 200
display_count = 0
error_count = 0
for i in range(total_tested):
    # Get random test image from valid classes only
    class_name = random.choice(classes)
    class_path = os.path.join(test_dir, class_name)

    try:
        # Get list of images in this class
        images = os.listdir(class_path)
        if len(images) == 0:
            error_count += 1
            continue
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)

        # Get prediction
        result = predict_and_display(img_path, best_model, class_names)

        # Track correctness
        if result['predicted_class'] == class_name:
            correct_predictions += 1
            status = "Right"
            color = 'green'
        else:
            status = 'worng'
            color = 'red'

        # Display first 20 images only
        if display_count < 20:
            axes[display_count].imshow(result['image'])
            title = f"{status} T:{class_name[:10]}\nP:{result['predicted_class'][:10]}\nC:{result['confidence']:.2f}"
            axes[display_count].set_title(title, color=color, fontsize=8)
            axes[display_count].axis('off')
            display_count += 1

        # Show progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total_tested} images...")

    except Exception as e:
        error_count += 1
        print(f"Error on image {i + 1}: {str(e)[:50]}...")
        continue

# Hide unused subplots
for j in range(display_count, len(axes)):
    axes[j].axis('off')

plt.suptitle(f'Sample Predictions (Showing {display_count} of {total_tested} images)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("TEST RESULTS SUMMARY")
print("=" * 60)
print(f"Total images attempted: {total_tested}")
print(f"Successful tests: {total_tested - error_count}")
print(f"Errors skipped: {error_count}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy on successful tests: {correct_predictions / (total_tested - error_count) * 100:.2f}%")
print("=" * 60)


print("="*60)
print("RESULTS SUMMARY AND MODEL SAVING")
print("="*60)

#Save the final model (using recommended .keras format)
final_model_path = '/kaggle/working/fruits360_final_model.keras'
best_model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

#Also save a backup in .h5 format if needed
h5_backup_path = '/kaggle/working/fruits360_final_model.h5'
best_model.save(h5_backup_path)
print(f"Backup model saved to: {h5_backup_path}")

#Check if history exists before trying to use it
try:
    #Save training history
    import json
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    with open('/kaggle/working/training_history.json', 'w') as f:
        json.dump(history_dict, f)
    print("Training history saved to: /kaggle/working/training_history.json")
    history_exists = True
except (NameError, AttributeError):
    print("Training history not available - skipping history save")
    history_exists = False

#Create comprehensive results summary
print("\n" + "="*60)
print("PROJECT FINAL RESULTS SUMMARY")
print("="*60)

#Get values safely
if history_exists:
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    epochs_completed = len(history.history['accuracy'])
else:
    final_train_acc = 0
    final_val_acc = 0
    epochs_completed = "N/A"

#FIXED: Calculate trainable parameters correctly
try:
    # Method 1: For TensorFlow 2.x
    trainable_params = sum([tf.keras.backend.count_params(v) for v in best_model.trainable_variables])
except:
    try:
        # Method 2: Alternative method
        trainable_params = sum([np.prod(v.shape) for v in best_model.trainable_variables])
    except:
        trainable_params = best_model.count_params()  # Fallback to total params

total_params = best_model.count_params()

summary = f"""
DATASET INFORMATION:
   • Dataset: Fruits-360
   • Image Size: {IMG_SIZE}x{IMG_SIZE}
   • Total Classes: {n_classes}
   • Training Samples: {train_generator.samples}
   • Test Samples: {test_generator.samples}

 MODEL ARCHITECTURE:
   • Type: Custom CNN with 4 Convolutional Blocks
   • Total Parameters: {total_params:,}
   • Trainable Parameters: {trainable_params:,}
   • Non-trainable Parameters: {total_params - trainable_params:,}

TRAINING CONFIGURATION:
   • Batch Size: {BATCH_SIZE}
   • Epochs: {epochs_completed}
   • Optimizer: Adam (lr=0.001)
   • Data Augmentation: Yes

PERFORMANCE METRICS:
   • Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
   • Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)
   • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
   • Test Loss: {test_loss:.4f}
   TARGET ACHIEVEMENT:
   • Target: >80% Test Accuracy
   • Achieved: {test_accuracy*100:.2f}%
   • Status: {' PASSED' if test_accuracy > 0.80 else '❌ NOT MET'}

   SAVED FILES:
   • Best Model: /kaggle/working/best_fruit_model.h5
   • Final Model (Keras): /kaggle/working/fruits360_final_model.keras
   • Final Model (H5 backup): /kaggle/working/fruits360_final_model.h5
   • Results Summary: /kaggle/working/results_summary.txt
"""

if history_exists:
    summary += "   • Training History: /kaggle/working/training_history.json\n"

print(summary)

# Save summary to file
with open('/kaggle/working/results_summary.txt', 'w') as f:
    f.write(summary)

print("\nAll files saved in /kaggle/working/:")
for file in os.listdir('/kaggle/working'):
    if file.endswith(('.h5', '.keras', '.json', '.txt')):
        file_size = os.path.getsize(f'/kaggle/working/{file}') / (1024*1024)  # Size in MB
        print(f"   • {file:35s} ({file_size:.2f} MB)")