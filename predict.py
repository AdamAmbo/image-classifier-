from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("cat_classifier_model.keras")

img_path = "test6.jpeg" 
img = image.load_img(img_path, target_size=(160, 160))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]


if prediction < 0.5:
    print(f" Cat (Confidence: {1 - prediction:.2f})")
else:
    print(f" Not a Cat (Confidence: {prediction:.2f})")
