#!/usr/bin/env python
# coding: utf-8

# In[1]:


# run this to see all the outputs below each cell, not only the last instruction
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # Rename images in img/ (remove _RGB)

# In[2]:


import os

IMG_DIR = "/media/nima/7CEEFEC0EEFE71AE/space/FLIR/img/"

for filename in os.listdir(IMG_DIR):
    if "_RGB" in filename:
        old_path = os.path.join(IMG_DIR, filename)
        new_filename = filename.replace("_RGB", "")
        new_path = os.path.join(IMG_DIR, new_filename)

        os.rename(old_path, new_path)

print("✅ Renaming in img/ completed.")


# # Rename + Convert images in imgr/ (_PreviewData + jpeg → jpg)

# In[3]:


import os
from PIL import Image

IMGR_DIR = "/media/nima/7CEEFEC0EEFE71AE/space/FLIR/imgr/"

for filename in os.listdir(IMGR_DIR):
    if not filename.lower().endswith(".jpeg"):
        continue

    old_path = os.path.join(IMGR_DIR, filename)

    # Remove "_PreviewData" and change extension to .jpg
    base_name = filename.replace("_PreviewData", "")
    base_name = os.path.splitext(base_name)[0]
    new_filename = base_name + ".jpg"
    new_path = os.path.join(IMGR_DIR, new_filename)

    # Open and save as JPG
    with Image.open(old_path) as img:
        img = img.convert("RGB")  # ensure compatibility
        img.save(new_path, "JPEG")

    # Remove original jpeg
    os.remove(old_path)

print("✅ Renaming and JPEG → JPG conversion in imgr/ completed.")


# # Rename TXT files in label/ (remove _PreviewData)

# In[1]:


import os

LABEL_DIR = "/media/nima/7CEEFEC0EEFE71AE/space/FLIR/label/"

for filename in os.listdir(LABEL_DIR):
    if filename.endswith(".txt") and "_PreviewData" in filename:
        old_path = os.path.join(LABEL_DIR, filename)

        new_filename = filename.replace("_PreviewData", "")
        new_path = os.path.join(LABEL_DIR, new_filename)

        os.rename(old_path, new_path)

print("✅ Renaming TXT files in label/ completed.")


# # line numbers

# In[2]:


# Replace 'file1.txt' and 'file2.txt' with your actual filenames
files = ['/media/nima/7CEEFEC0EEFE71AE/space/FLIR/align_train.txt', 
         '/media/nima/7CEEFEC0EEFE71AE/space/FLIR/align_validation.txt']

for filename in files:
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Remove "_PreviewData" from each line and strip newline just in case
    lines = [line.replace('_PreviewData', '').strip() + '\n' for line in lines]

    # Write back to the same file
    with open(filename, 'w') as f:
        f.writelines(lines)


# # train_val split

# In[1]:


import os
import shutil

BASE_DIR = "/media/nima/7CEEFEC0EEFE71AE/space/FLIR"

SPLITS = {
    "train": "align_train.txt",
    "val": "align_validation.txt",
}

SRC_DIRS = {
    "imgr": os.path.join(BASE_DIR, "imgr"),
    "img": os.path.join(BASE_DIR, "img"),
    "label": os.path.join(BASE_DIR, "label"),
}

DST_DIRS = {
    "train": {
        "imgr": os.path.join(BASE_DIR, "train", "imgr"),
        "img": os.path.join(BASE_DIR, "train", "img"),
        "label": os.path.join(BASE_DIR, "train", "label"),
    },
    "val": {
        "imgr": os.path.join(BASE_DIR, "val", "imgr"),
        "img": os.path.join(BASE_DIR, "val", "img"),
        "label": os.path.join(BASE_DIR, "val", "label"),
    },
}

# Create destination directories if they don't exist
for split in DST_DIRS:
    for d in DST_DIRS[split].values():
        os.makedirs(d, exist_ok=True)

def copy_split(split_name, align_file):
    with open(os.path.join(BASE_DIR, align_file), "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    for name in filenames:
        # Infrared image
        shutil.copy(
            os.path.join(SRC_DIRS["imgr"], f"{name}.jpg"),
            os.path.join(DST_DIRS[split_name]["imgr"], f"{name}.jpg")
        )

        # Visible image
        shutil.copy(
            os.path.join(SRC_DIRS["img"], f"{name}.jpg"),
            os.path.join(DST_DIRS[split_name]["img"], f"{name}.jpg")
        )

        # Label
        shutil.copy(
            os.path.join(SRC_DIRS["label"], f"{name}.txt"),
            os.path.join(DST_DIRS[split_name]["label"], f"{name}.txt")
        )

    print(f"{split_name} split: copied {len(filenames)} samples")

# Run for train and validation
for split, align_file in SPLITS.items():
    copy_split(split, align_file)

