# Step 1: Download using Kaggle API
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d badasstechie/celebahq-resized-256x256
!unzip celebahq-resized-256x256.zip -d celebahq-resized-256x256

# Step 2: Split into train and valid only
import os, random, shutil

image_dir = "/content/celebahq-resized-256x256/celeba_hq_256"
train_dir = "/content/celebahq-resized-256x256/train"
valid_dir = "/content/celebahq-resized-256x256/valid"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
random.shuffle(image_files)

train_split, valid_split = 2000, 500

for i, image_file in enumerate(image_files[:train_split + valid_split]):
    source = os.path.join(image_dir, image_file)
    dest = train_dir if i < train_split else valid_dir
    shutil.copy2(source, os.path.join(dest, image_file))
# Step 3: Random cropping to 128x128 (HR)
from PIL import Image
import os

def random_crop_resize(image_path, output_dir, size=(128, 128)):
    img = Image.open(image_path)
    w, h = img.size
    left = random.randint(0, w - size[0])
    top = random.randint(0, h - size[1])
    cropped = img.crop((left, top, left + size[0], top + size[1]))
    cropped.save(os.path.join(output_dir, os.path.basename(image_path)))

cropped_train_dir = "/content/celebahq-resized-256x256/hr_train"
cropped_valid_dir = "/content/celebahq-resized-256x256/hr_valid"
os.makedirs(cropped_train_dir, exist_ok=True)
os.makedirs(cropped_valid_dir, exist_ok=True)

for filename in os.listdir(train_dir):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        random_crop_resize(os.path.join(train_dir, filename), cropped_train_dir)

for filename in os.listdir(valid_dir):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        random_crop_resize(os.path.join(valid_dir, filename), cropped_valid_dir)
# Step 4: Resize HR (128x128) to LR (32x32) using bicubic
resized_train_dir = "/content/celebahq-resized-256x256/lr_train"
resized_valid_dir = "/content/celebahq-resized-256x256/lr_valid"
os.makedirs(resized_train_dir, exist_ok=True)
os.makedirs(resized_valid_dir, exist_ok=True)

def resize_bicubic(input_dir, output_dir, size=(32, 32)):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            img = Image.open(os.path.join(input_dir, filename))
            img = img.resize(size, Image.BICUBIC)
            img.save(os.path.join(output_dir, filename))

resize_bicubic(cropped_train_dir, resized_train_dir)
resize_bicubic(cropped_valid_dir, resized_valid_dir)
# Step 5: Convert to NumPy arrays and save
import numpy as np

def images_to_numpy(image_dir):
    data = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            try:
                img = Image.open(os.path.join(image_dir, filename)).convert("RGB")
                data.append(np.array(img))
            except:
                continue
    return np.array(data)

hr_train = images_to_numpy(cropped_train_dir)
lr_train = images_to_numpy(resized_train_dir)
hr_valid = images_to_numpy(cropped_valid_dir)
lr_valid = images_to_numpy(resized_valid_dir)

np.save('/content/hr_train.npy', hr_train)
np.save('/content/lr_train.npy', lr_train)
np.save('/content/hr_valid.npy', hr_valid)
np.save('/content/lr_valid.npy', lr_valid)

