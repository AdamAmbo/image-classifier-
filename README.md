
## Cat vs Not-a-Cat Classifier

This project trains and evaluates a deep learning model that classifies images as **Cat** or **Not a Cat**.  

---
## How It Works
---

- **organize_data.py** → organizes the dataset into `train/` and `val/` folders, each containing `cat/` and `not_cat/` images.
- **ai.py** → builds and trains the CNN model on the prepared data, then saves it as `cat_classifier_model.keras`.
- **predict.py** → loads the trained model and makes predictions on new images, outputting class, score, and confidence.
---

## Project Structure
```

├── README.md
├── ai.py → training script (builds & trains the CNN model, saves as `.keras`)
├── organize_data.py → prepares dataset folders (`train/` and `val/` splits)
├── predict.py → loads the trained model and makes predictions on new images
└── cat_classifier_model.keras → saved trained model (Keras 3 format)
```
---

## Setup

1. **Clone this repo** (or download the files).
2. **Install requirements**:
   ```bash
   pip install tensorflow>=2.15,<3.0 numpy


> If using GPU, install the appropriate TensorFlow build for your system.

3. **Download the dataset**:

   * Original Kaggle Cats vs Dogs dataset: https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
   

---

## Data Preparation

Run the organizer script to create train/val splits:

```bash
python organize_data.py
```

This will generate a `data/` folder:

```
data/
 ├── train/
 │    ├── cat/
 │    └── not_cat/
 └── val/
      ├── cat/
      └── not_cat/
```

---

## Training

Train the CNN and save the model:

```bash
python ai.py
```

The trained model will be saved as:

```
cat_classifier_model.keras
```

---

## Prediction

Run predict.py on a new image:

```bash
img_path = "test.jpeg" 
```

Output example:

```
Prediction: Cat
Raw score (sigmoid): 0.9234
Confidence: 92.34% (threshold = 0.50)
```

Options:

* `--threshold` → set custom decision threshold (default: 0.5)

---

## Next Steps

* Add evaluation script for accuracy/precision/recall on `val/`
* Try different CNN architectures (e.g., MobileNetV2, ResNet50)
* Experiment with data augmentation for better generalization

---

## License

This project is released under the MIT License.

