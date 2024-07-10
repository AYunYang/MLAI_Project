import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
original_dataset_dir = 'C:/Users/Yun Yang/OneDrive/Desktop/MLAI_Images'
base_dir = 'datasets'
classes = ['eggtart','salmonsashimi', 'unknown']

# Create directories for train, validation, and test splits
os.makedirs(base_dir, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

for class_name in classes:
    # Create class directories in train, validation, and test folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Get list of all images in the class directory
    class_dir = os.path.join(original_dataset_dir, class_name)
    images = os.listdir(class_dir)
    
    # Split images into train, validation, and test sets (80%, 10%, 10%)
    train_images, temp_images = train_test_split(images, test_size=0.2, random_state=42)
    validation_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    # Copy images to corresponding directories
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copyfile(src, dst)
    
    for img in validation_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(validation_dir, class_name, img)
        shutil.copyfile(src, dst)

    for img in test_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(test_dir, class_name, img)
        shutil.copyfile(src, dst)

print("Data successfully split into train, validation, and test sets.")

# Verify the split by counting the number of images in each directory
def count_files(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])

print("Training set size:", count_files(train_dir))
print("Validation set size:", count_files(validation_dir))
print("Test set size:", count_files(test_dir))