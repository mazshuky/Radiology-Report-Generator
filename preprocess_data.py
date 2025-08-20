import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import os
import glob

base_path = "dataset" # Adjust to dataset path

# Load metadata CSV
data = pd.read_csv(os.path.join(base_path, "Data_Entry_2017.csv"))
print(data.head())

with open(os.path.join(base_path, "train_val_list.txt")) as f:
    train_val_list = f.read().splitlines()

with open(os.path.join(base_path, "test_list.txt")) as f:
    test_list = f.read().splitlines()

print("Train/Val count:", len(train_val_list))
print("Test count:", len(test_list))

# Filter data for train/val and test sets
train_val_df = data[data["Image Index"].isin(train_val_list)].copy()
test_df = data[data["Image Index"].isin(test_list)].copy()

print("Train/Val DF shape:", train_val_df.shape)
print("Test DF shape:", test_df.shape)


def create_image_path_mapping():
    """Create a mapping of image filenames to their full paths"""
    image_paths = {}

    # Search in all images_xxx/images/ subdirectories
    images_dirs = glob.glob(os.path.join(base_path, "images", "images_*", "images"))

    print(f"Found {len(images_dirs)} image directories")

    for images_dir in images_dirs:
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.endswith('.png'):
                    image_paths[img_file] = os.path.join(images_dir, img_file)

    print(f"Total images found: {len(image_paths)}")
    return image_paths


# Create image path mapping
image_path_mapping = create_image_path_mapping()


def filter_existing_images(df):
    """Filter dataframe to only include images that exist in our mapping"""
    existing_mask = df["Image Index"].isin(image_path_mapping.keys())
    filtered_df = df[existing_mask].copy()

    print(f"Original samples: {len(df)}")
    print(f"Existing images: {len(filtered_df)}")
    print(f"Missing images: {len(df) - len(filtered_df)}")

    return filtered_df


# Filter out missing images
train_val_df = filter_existing_images(train_val_df)
test_df = filter_existing_images(test_df)

# Split labels by '|'
train_val_df["Finding Labels"] = train_val_df["Finding Labels"].apply(lambda x: x.split('|'))
test_df["Finding Labels"] = test_df["Finding Labels"].apply(lambda x: x.split('|'))

# Multi-label binarizer (one-hot encoding for all diseases)
mlb = MultiLabelBinarizer()
y_train_val = mlb.fit_transform(train_val_df["Finding Labels"])
y_test = mlb.transform(test_df["Finding Labels"])

print("Classes:", mlb.classes_)
print("Shape of label matrix:", y_train_val.shape)

IMG_SIZE = 224 # Resize images to this size


def load_image(file, size=IMG_SIZE):
    """Load image using the path mapping"""
    if file not in image_path_mapping:
        print(f"Warning: Image not found in mapping: {file}")
        return None

    path = image_path_mapping[file]
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image: {path}")
        return None

    img = cv2.resize(img, (size, size))
    img = img.astype("float32") / 255.0
    return img


# Test with first available sample
if len(train_val_df) > 0:
    sample_file = train_val_df.iloc[0]["Image Index"]
    sample_img = load_image(sample_file)

    if sample_img is not None:
        print("Sample image shape:", sample_img.shape)
        print(f"Sample image path: {image_path_mapping[sample_file]}")
    else:
        print("Could not load sample image")
else:
    print("No valid training samples found")