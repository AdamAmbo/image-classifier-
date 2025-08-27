import os
import shutil
import random

# Source folders
source_base = 'raw/kagglecatsanddogs_3367a/PetImages'
target_base = 'data'
categories = ['cat', 'not_cat']  

val_split = 0.2

for category in categories:
    source_folder = os.path.join(source_base, 'Cat' if category == 'cat' else 'Dog')
    all_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
    random.shuffle(all_files)

    val_count = int(len(all_files) * val_split)
    train_files = all_files[val_count:]
    val_files = all_files[:val_count]

    for group, files in [('train', train_files), ('val', val_files)]:
        dest_folder = os.path.join(target_base, group, category)
        os.makedirs(dest_folder, exist_ok=True)

        for file in files:
            src = os.path.join(source_folder, file)
            dst = os.path.join(dest_folder, file)
            try:
                shutil.copy2(src, dst)
            except:
                pass  # skip unreadable/corrupted files

print(" Dataset organized into 'data/train/' and 'data/val/'")
