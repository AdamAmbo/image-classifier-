import os
import shutil
import random
# =========================
# CONFIGURATION
# =========================
source_base = 'raw/kagglecatsanddogs_3367a/PetImages'         # Path to raw Kaggle dataset
target_base = 'data'                                          # Target folder for organized dataset
categories = ['cat', 'not_cat']                               # Final class names in our dataset

val_split = 0.2          # 20% of images go to validation, 80% to training



# =========================
# DATA ORGANIZATION
# =========================
for category in categories:
    # Match our categories ("cat"/"not_cat") to Kaggle folder names ("Cat"/"Dog")
    source_folder = os.path.join(source_base, 'Cat' if category == 'cat' else 'Dog')

    # Collect all JPG files for the category
    all_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]

    # Shuffle to ensure random train/val split
    random.shuffle(all_files)
    
    # Split into training and validation sets
    val_count = int(len(all_files) * val_split)
    train_files = all_files[val_count:]
    val_files = all_files[:val_count]

    # =========================
    # COPY FILES TO NEW STRUCTURE
    # =========================
    for group, files in [('train', train_files), ('val', val_files)]:
        # Destination folder: e.g. data/train/cat, data/val/not_cat
        dest_folder = os.path.join(target_base, group, category)
        os.makedirs(dest_folder, exist_ok=True)
        
         # Copy files one by one
        for file in files:
            src = os.path.join(source_folder, file)
            dst = os.path.join(dest_folder, file)
            try:
                shutil.copy2(src, dst)
            except:
                pass  # skip unreadable/corrupted files

print(" Dataset organized into 'data/train/' and 'data/val/'")

