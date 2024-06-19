import os
import random
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import functional as F

# Function to augment an image using torchvision transforms
def augment_image(image):
    augmentations = [
        transforms.RandomRotation(degrees=90),
        transforms.RandomRotation(degrees=180),
        transforms.RandomRotation(degrees=270),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    ]
    transform = transforms.Compose([
        random.choice(augmentations),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    return transform(image)

# Function to balance images in each folder
def balance_images(base_path):
    for root, dirs, files in os.walk(base_path):
        if dirs:
            subdir_max_images = {}
            # Traverse each subdir group to find the max images
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                image_count = len([f for f in os.listdir(dir_path) if f.endswith(('jpg', 'png', 'jpeg'))])
                subdir_max_images[dir] = image_count

            max_images = max(subdir_max_images.values())

            for dir in dirs:
                dir_path = os.path.join(root, dir)
                image_files = [f for f in os.listdir(dir_path) if f.endswith(('jpg', 'png', 'jpeg'))]
                current_count = len(image_files)
                augment_count = max_images - current_count

                if augment_count > 0:
                    print(f"Augmenting {augment_count} images in {dir_path}")
                    for i in range(augment_count):
                        image_to_augment = random.choice(image_files)
                        image_path = os.path.join(dir_path, image_to_augment)
                        image = Image.open(image_path)
                        augmented_image = augment_image(image)
                        new_image_name = f"{os.path.splitext(image_to_augment)[0]}_aug_{i}{os.path.splitext(image_to_augment)[1]}"
                        augmented_image.save(os.path.join(dir_path, new_image_name))

# Base directory path containing all the folders
base_directory = "/home/srikanth/Interns/RGB_images"
balance_images(base_directory)
