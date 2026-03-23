# Import necessary libraries
import kagglehub
import tensorflow as tf
import json
import os
import random
from collections import Counter

from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR    = r'D:\Master\Sem2\ACABI\Fruits_classifier'
SKIP_TRAINING = True   # Set to False to retrain from scratch
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MODEL DEFINITION
# ============================================================
def create_cnn_model(input_shape=(100, 100, 3), num_classes=None):
    """
    Custom CNN architecture for fruit classification.
    - 4 Convolutional blocks with increasing filters
    - Batch Normalization after each conv layer
    - MaxPooling for dimensionality reduction
    - Dropout for regularization
    - Global Average Pooling instead of Flatten
    - Dense layers with dropout for classification
    """
    model = models.Sequential([
        # 1st convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape, name='conv1_1'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2, name='pool1'),
        layers.Dropout(0.25, name='dropout1'),

        # 2nd convolutional block
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

        # 4th convolutional block
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
    """Predict class for a single image and return results."""
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions   = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence    = predictions[0][predicted_idx]

    top_3_idx         = np.argsort(predictions[0])[-3:][::-1]
    top_3_classes     = [class_names[i] for i in top_3_idx]
    top_3_confidences = [predictions[0][i] for i in top_3_idx]

    return {
        'image':             img,
        'predicted_class':   class_names[predicted_idx],
        'confidence':        confidence,
        'top_3_classes':     top_3_classes,
        'top_3_confidences': top_3_confidences
    }


# ============================================================
# DOWNLOAD DATASET
# ============================================================
path = kagglehub.dataset_download("moltean/fruits")
path = os.path.join(path, 'fruits-360_100x100', 'fruits-360')
print("Path to dataset files:", path)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ============================================================
# CHECK DATASET STRUCTURE
# ============================================================
print("\n" + "=" * 60)
print("CHECKING DATASET STRUCTURE")
print("=" * 60)

print("\nContents of dataset path:")
for item in os.listdir(path):
    print(f"- {item}")
    item_path = os.path.join(path, item)
    if os.path.isdir(item_path):
        for subitem in os.listdir(item_path)[:5]:
            print(f"  {subitem}")

# Find dataset path with Training and Test folders
dataset_path = None
if os.path.exists(os.path.join(path, 'Training')) and os.path.exists(os.path.join(path, 'Test')):
    dataset_path = path
    print(f"\nFound dataset at: {path}")
    print("   ✓ Contains Training and Test folders")
else:
    for item in os.listdir(path):
        subpath = os.path.join(path, item)
        if os.path.isdir(subpath) and os.path.exists(os.path.join(subpath, 'Training')):
            dataset_path = subpath
            print(f"\n   ✓ Found Training in: {subpath}")
            break

if dataset_path is None:
    raise FileNotFoundError("Could not find Training/Test folders in downloaded dataset.")

train_dir = os.path.join(dataset_path, 'Training')
test_dir  = os.path.join(dataset_path, 'Test')

print(f"\nTraining directory: {train_dir}")
print(f"Test directory:     {test_dir}")
print(f"Training exists: {os.path.exists(train_dir)}")
print(f"Test exists:     {os.path.exists(test_dir)}")

# ============================================================
# EXPLORE DATASET
# ============================================================
print("=" * 60)
print("EXPLORING DATASET")
print("=" * 60)

classes   = sorted(os.listdir(train_dir))
n_classes = len(classes)
print(f"\nTotal number of classes: {n_classes}")
print(f"First 10 classes: {classes[:10]}")
print(f"Last 10 classes:  {classes[-10:]}")

print("\nDisplaying sample images from dataset...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()
for i in range(10):
    class_name = random.choice(classes)
    class_path = os.path.join(train_dir, class_name)
    images = [f for f in os.listdir(class_path)
              if os.path.isfile(os.path.join(class_path, f))
              and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        continue
    img_path = os.path.join(class_path, random.choice(images))
    img = plt.imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"{class_name}\n{img.shape}", fontsize=10, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Sample Images from Fruits-360 Dataset', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# DATA PREPROCESSING AND AUGMENTATION
# ============================================================
print("=" * 60)
print("DATA PREPROCESSING AND AUGMENTATION")
print("=" * 60)

IMG_SIZE   = 100
BATCH_SIZE = 32
EPOCHS     = 30

print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
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

print("\nData generators created successfully!")
print(f"Training samples:  {train_generator.samples}")
print(f"Test samples:      {test_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")

class_names = list(train_generator.class_indices.keys())

# ============================================================
# BUILD MODEL
# ============================================================
print("=" * 60)
print("BUILDING CUSTOM CNN MODEL")
print("=" * 60)

model = create_cnn_model(num_classes=n_classes)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled successfully!")

# ============================================================
# CALLBACKS
# ============================================================
best_model_path = os.path.join(OUTPUT_DIR, 'best_fruit_model.h5')

callbacks = [
    EarlyStopping(
        monitor='loss',
        patience=7,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    ),
    ModelCheckpoint(
        filepath=best_model_path,
        monitor='accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# ============================================================
# TRAIN OR LOAD MODEL
# ============================================================
print("=" * 60)
print("LOADING SAVED MODEL" if SKIP_TRAINING else "TRAINING THE MODEL")
print("=" * 60)

if SKIP_TRAINING:
    print(f"Skipping training — loading saved model from:\n  {best_model_path}")
    best_model = load_model(best_model_path)
    print("Model loaded successfully!")
    final_train_acc  = 'N/A (loaded from saved model)'
    epochs_completed = 'N/A (loaded from saved model)'
else:
    print(f"Epochs: {EPOCHS}  |  Batch Size: {BATCH_SIZE}  |  Training samples: {train_generator.samples}")
    print("=" * 60)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    print("\nTraining completed!")
    best_model = load_model(best_model_path)

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    final_train_acc  = history.history['accuracy'][-1]
    epochs_completed = len(history.history['accuracy'])
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final training loss:     {history.history['loss'][-1]:.4f}")

    # Save training history
    history_json_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'loss':     [float(x) for x in history.history['loss']],
    }
    with open(history_json_path, 'w') as f:
        json.dump(history_dict, f)
    print(f"Training history saved to: {history_json_path}")

# ============================================================
# EVALUATE MODEL
# ============================================================
print("=" * 60)
print("EVALUATING MODEL ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=1)

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"   Test Loss:         {test_loss:.4f}")
print(f"   Test Accuracy:     {test_accuracy:.4f}")
print(f"   Test Accuracy (%): {test_accuracy * 100:.2f}%")

if test_accuracy > 0.80:
    print(f"\n SUCCESS! Model achieved {test_accuracy * 100:.2f}% — target of >80% MET!")
else:
    print(f"\n Model achieved {test_accuracy * 100:.2f}% — {(80 - test_accuracy * 100):.2f}% away from target.")

# ============================================================
# DETAILED PERFORMANCE ANALYSIS
# ============================================================
print("=" * 60)
print("DETAILED PERFORMANCE ANALYSIS")
print("=" * 60)

test_generator.reset()
predictions       = best_model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes      = test_generator.classes

print(f"\nNumber of classes in test set: {len(np.unique(true_classes))}")

# Full classification report
print("\nFull Classification Report (all classes):")
print("-" * 80)
print(classification_report(true_classes, predicted_classes,
                             target_names=class_names, zero_division=0))

# First 15 classes report
print("=" * 60)
print("Classification Report (First 15 Classes Only):")
print("=" * 60)
first_15_indices     = list(range(min(15, len(class_names))))
first_15_class_names = [class_names[i] for i in first_15_indices]
mask          = np.isin(true_classes, first_15_indices)
filtered_true = true_classes[mask]
filtered_pred = predicted_classes[mask]

print(f"Total samples in first 15 classes: {len(filtered_true)}")
if len(filtered_true) > 0:
    print(classification_report(filtered_true, filtered_pred,
                                 labels=first_15_indices,
                                 target_names=first_15_class_names,
                                 zero_division=0))
else:
    print("No samples found for first 15 classes in test set!")

# Summary statistics
print("=" * 60)
print("PERFORMANCE SUMMARY STATISTICS")
print("=" * 60)

accuracy = np.mean(predicted_classes == true_classes)
print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

print(f"\nPer-class accuracy (First {min(15, len(class_names))} Classes):")
print("-" * 60)
print(f"{'Class Name':<30} {'Samples':<10} {'Correct':<10} {'Accuracy':<10}")
print("-" * 60)
for i in range(min(15, len(class_names))):
    cname    = class_names[i]
    cmask    = (true_classes == i)
    csamples = np.sum(cmask)
    if csamples > 0:
        ccorrect = np.sum(predicted_classes[cmask] == i)
        cacc     = ccorrect / csamples
        print(f"{cname:<30} {csamples:<10} {ccorrect:<10} {cacc * 100:<10.2f}%")
    else:
        print(f"{cname:<30} {'0':<10} {'0':<10} {'N/A':<10}")

# Best and worst performing classes
class_accuracies = []
for i in range(len(class_names)):
    cmask = (true_classes == i)
    if np.sum(cmask) > 5:
        cacc = np.mean(predicted_classes[cmask] == i)
        class_accuracies.append((class_names[i], cacc, np.sum(cmask)))

class_accuracies.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 Best Performing Classes:")
for i, (name, acc, samples) in enumerate(class_accuracies[:5]):
    print(f"   {i+1}. {name:<30} Accuracy: {acc*100:.2f}% (Samples: {samples})")

print("\nBottom 5 Worst Performing Classes:")
for i, (name, acc, samples) in enumerate(class_accuracies[-5:]):
    print(f"   {i+1}. {name:<30} Accuracy: {acc*100:.2f}% (Samples: {samples})")

# Confusion matrix (first 15 classes)
print("\nGenerating Confusion Matrix (First 15 classes)...")
first_15_mask = np.isin(true_classes, first_15_indices)
cm_true = true_classes[first_15_mask]
cm_pred = predicted_classes[first_15_mask]

if len(cm_true) > 0:
    cm = confusion_matrix(cm_true, cm_pred, labels=first_15_indices)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=first_15_class_names,
                yticklabels=first_15_class_names,
                annot_kws={'size': 10})
    plt.title('Confusion Matrix (First 15 Classes)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    print("\nConfusion Matrix Statistics:")
    row_sums      = cm.sum(axis=1)
    per_class_acc = cm.diagonal() / (row_sums + 1e-10)
    for i, cname in enumerate(first_15_class_names):
        if row_sums[i] > 0:
            print(f"   {cname:<30} Accuracy: {per_class_acc[i]*100:.2f}%")
else:
    print("No samples from first 15 classes found — showing top 15 most frequent instead.")
    class_counts     = Counter(true_classes)
    top_15_classes   = [cls for cls, _ in class_counts.most_common(15)]
    top_15_names     = [class_names[cls] for cls in top_15_classes]
    top_15_mask      = np.isin(true_classes, top_15_classes)
    cm_true_top      = true_classes[top_15_mask]
    cm_pred_top      = predicted_classes[top_15_mask]
    remap_dict       = {old: new for new, old in enumerate(top_15_classes)}
    cm_true_remapped = np.array([remap_dict[x] for x in cm_true_top])
    cm_pred_remapped = np.array([remap_dict[x] for x in cm_pred_top])
    cm_top = confusion_matrix(cm_true_remapped, cm_pred_remapped, labels=range(15))
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_15_names, yticklabels=top_15_names,
                annot_kws={'size': 10})
    plt.title('Confusion Matrix (Top 15 Most Frequent Classes)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# ============================================================
# TESTING ON 10 SAMPLE IMAGES
# ============================================================
print("=" * 60)
print("TESTING ON 10 SAMPLE IMAGES")
print("=" * 60)

valid_test_classes = [c for c in sorted(os.listdir(train_dir))
                      if os.path.isdir(os.path.join(test_dir, c))]
print(f"Valid test classes: {len(valid_test_classes)}")

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.ravel()
correct_predictions = 0

for i in range(10):
    class_name = random.choice(valid_test_classes)
    class_path = os.path.join(test_dir, class_name)
    img_files  = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        continue
    img_path = os.path.join(class_path, random.choice(img_files))
    result   = predict_and_display(img_path, best_model, class_names)

    axes[i].imshow(result['image'])
    is_correct = result['predicted_class'] == class_name
    if is_correct:
        correct_predictions += 1
    color  = 'green' if is_correct else 'red'
    status = 'CORRECT' if is_correct else 'WRONG'
    title  = (f"True: {class_name[:15]}\n"
              f"Pred: {result['predicted_class'][:15]}\n"
              f"Conf: {result['confidence']:.2f}\n"
              f"{status}")
    axes[i].set_title(title, color=color, fontsize=9, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Model Predictions on Test Images (Green=Correct, Red=Incorrect)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
print(f"\nCorrect predictions: {correct_predictions}/10  ({correct_predictions * 10:.1f}%)")

# ============================================================
# TESTING ON 200 SAMPLE IMAGES
# ============================================================
print("=" * 60)
print("TESTING ON 200 SAMPLE IMAGES")
print("=" * 60)

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.ravel()
correct_predictions = 0
total_tested        = 200
display_count       = 0
error_count         = 0

for i in range(total_tested):
    class_name = random.choice(valid_test_classes)
    class_path = os.path.join(test_dir, class_name)
    try:
        img_files = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            error_count += 1
            continue
        img_path = os.path.join(class_path, random.choice(img_files))
        result   = predict_and_display(img_path, best_model, class_names)

        is_correct = result['predicted_class'] == class_name
        if is_correct:
            correct_predictions += 1
        color  = 'green' if is_correct else 'red'
        status = 'Right' if is_correct else 'Wrong'

        if display_count < 20:
            axes[display_count].imshow(result['image'])
            title = (f"{status} T:{class_name[:10]}\n"
                     f"P:{result['predicted_class'][:10]}\n"
                     f"C:{result['confidence']:.2f}")
            axes[display_count].set_title(title, color=color, fontsize=8)
            axes[display_count].axis('off')
            display_count += 1

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total_tested} images...")

    except Exception as e:
        error_count += 1
        print(f"Error on image {i + 1}: {str(e)[:50]}...")

for j in range(display_count, len(axes)):
    axes[j].axis('off')

plt.suptitle(f'Sample Predictions (Showing {display_count} of {total_tested} images)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

successful = total_tested - error_count
print("\n" + "=" * 60)
print("200-IMAGE TEST RESULTS SUMMARY")
print("=" * 60)
print(f"Total attempted:  {total_tested}")
print(f"Successful tests: {successful}")
print(f"Errors skipped:   {error_count}")
print(f"Correct:          {correct_predictions}")
print(f"Accuracy:         {correct_predictions / successful * 100:.2f}%")

# ============================================================
# SAVE MODELS AND RESULTS
# ============================================================
print("=" * 60)
print("RESULTS SUMMARY AND MODEL SAVING")
print("=" * 60)

final_model_path = os.path.join(OUTPUT_DIR, 'fruits360_final_model.keras')
h5_backup_path   = os.path.join(OUTPUT_DIR, 'fruits360_final_model.h5')
summary_txt_path = os.path.join(OUTPUT_DIR, 'results_summary.txt')

best_model.save(final_model_path)
print(f"Final model saved to:  {final_model_path}")
best_model.save(h5_backup_path)
print(f"Backup model saved to: {h5_backup_path}")

# Calculate parameters
try:
    trainable_params = sum([tf.keras.backend.count_params(v)
                            for v in best_model.trainable_variables])
except Exception:
    trainable_params = sum([np.prod(v.shape)
                            for v in best_model.trainable_variables])

total_params = best_model.count_params()

summary = f"""
DATASET INFORMATION:
   • Dataset:           Fruits-360
   • Image Size:        {IMG_SIZE}x{IMG_SIZE}
   • Total Classes:     {n_classes}
   • Training Samples:  {train_generator.samples}
   • Test Samples:      {test_generator.samples}

MODEL ARCHITECTURE:
   • Type:                     Custom CNN with 4 Convolutional Blocks
   • Total Parameters:         {total_params:,}
   • Trainable Parameters:     {trainable_params:,}
   • Non-trainable Parameters: {total_params - trainable_params:,}

TRAINING CONFIGURATION:
   • Batch Size:        {BATCH_SIZE}
   • Epochs Completed:  {epochs_completed}
   • Optimizer:         Adam (lr=0.001)
   • Data Augmentation: Yes

PERFORMANCE METRICS:
   • Final Training Accuracy: {final_train_acc}
   • Test Accuracy:           {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
   • Test Loss:               {test_loss:.4f}

TARGET ACHIEVEMENT:
   • Target:   >80% Test Accuracy
   • Achieved: {test_accuracy*100:.2f}%
   • Status:   {'PASSED' if test_accuracy > 0.80 else 'NOT MET'}

SAVED FILES:
   • Best Model (H5):         {best_model_path}
   • Final Model (Keras):     {final_model_path}
   • Final Model (H5 backup): {h5_backup_path}
   • Results Summary:         {summary_txt_path}
"""

print(summary)
with open(summary_txt_path, 'w') as f:
    f.write(summary)

print(f"\nAll files saved in: {OUTPUT_DIR}")
for file in os.listdir(OUTPUT_DIR):
    if file.endswith(('.h5', '.keras', '.json', '.txt')):
        size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, file)) / (1024 * 1024)
        print(f"   • {file:40s} ({size_mb:.2f} MB)")