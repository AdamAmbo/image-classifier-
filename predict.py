from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# =========================
# LOAD TRAINED MODEL
# =========================
# Load the trained cat vs not-cat classifier
model = tf.keras.models.load_model("cat_classifier_model.keras")

# =========================
# LOAD & PREPROCESS IMAGE
# =========================
img_path = "test6.jpeg" # Path to the test image
# Load and resize the image to match training input size
img = image.load_img(img_path, target_size=(160, 160))

# Convert image to numpy array and normalize pixel values to [0,1]
img_array = image.img_to_array(img) / 255.0

# Add an extra batch dimension (model expects [batch, height, width, channels])
img_array = np.expand_dims(img_array, axis=0)

# =========================
# MAKE PREDICTION
# =========================
# Model outputs a probability (sigmoid): closer to 0 = Cat, closer to 1 = Not a Cat
prediction = model.predict(img_array)[0][0]

# =========================
# DISPLAY RESULT
# =========================
if prediction < 0.5:
    # If probability < 0.5 → classify as Cat
    print(f" Cat (Confidence: {1 - prediction:.2f})")
else:
    # If probability >= 0.5 → classify as Not a Cat
    print(f" Not a Cat (Confidence: {prediction:.2f})")

