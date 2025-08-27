import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# ========== CONFIG ==========
IMG_SIZE = (160, 160) # Resize all images to 160x160 pixels
BATCH_SIZE = 32   # Number of images per training batch
EPOCHS = 20 # Training epochs
TRAIN_DIR = 'data/train' # Path to training dataset
VAL_DIR = 'data/val'   # Path to validation dataset
MODEL_PATH = 'cat_classifier_model.keras' # Where to save trained model

# ========== DATA LOADING ==========
# Load training dataset from directory structure:
# data/train/cat and data/train/not_cat
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    label_mode='binary',  # Binary classification (0 or 1)
    image_size=IMG_SIZE,  # Resize to 160x160
    batch_size=BATCH_SIZE,
    class_names=['cat', 'not_cat']  #  Enforce correct label mapping
)
# Same for validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=['cat', 'not_cat']  #  Enforce same order
)


# ========== DATA PIPELINE ==========

AUTOTUNE = tf.data.AUTOTUNE
# Random image augmentations to improve generalization
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
# Normalize pixel values to [0,1]
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
# Shuffle, cache, and prefetch for performance
train_ds = train_ds.map(preprocess).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

# ========== CLASS WEIGHTS ==========
# Count number of cat and not_cat images
cat_count = len(os.listdir(os.path.join(TRAIN_DIR, 'cat')))
not_cat_count = len(os.listdir(os.path.join(TRAIN_DIR, 'not_cat')))

# Build label list for weight calculation
labels = [0] * cat_count + [1] * not_cat_count

# Compute weights to balance classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {0: weights[0], 1: weights[1]}

# ========== MODEL ==========
model = tf.keras.Sequential([
    data_augmentation,                 # Apply random augments
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=IMG_SIZE + (3,)),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),         # Prevent overfitting
    tf.keras.layers.Dense(1, activation='sigmoid'),          # Single output: [0,1]
])
# Compile with Adam optimizer + Binary Crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ========== TRAINING ==========
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights) # Balance cats vs not-cats

# ========== SAVE ==========
model.save(MODEL_PATH)
print(f" Model saved as {MODEL_PATH}")

# ========== OPTIONAL: PLOT ==========
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

