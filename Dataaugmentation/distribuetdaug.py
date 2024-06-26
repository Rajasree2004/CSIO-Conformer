# import os
# import random
# from PIL import Image
# from torchvision import transforms

# # Function to augment an image using torchvision transforms
# def augment_image(image):
#     augmentations = [
#         transforms.RandomRotation(degrees=90),
#         transforms.RandomRotation(degrees=180),
#         transforms.RandomRotation(degrees=270),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
#     ]
#     transform = transforms.Compose([
#         random.choice(augmentations),
#         transforms.ToTensor(),
#         transforms.ToPILImage()
#     ])
#     return transform(image)

# # Calculate additional images required per subfolder
# def calculate_additional_images(subfolder_counts, total_to_add):
#     subfolder_names = list(subfolder_counts.keys())
#     subfolder_images_needed = {subfolder: 0 for subfolder in subfolder_names}
    
#     total_current_images = sum(subfolder_counts.values())
#     total_subfolders = len(subfolder_names)
    
#     # Distribute images as evenly as possible
#     per_folder_addition = total_to_add // total_subfolders
#     remainder = total_to_add % total_subfolders
    
#     for subfolder in subfolder_names:
#         subfolder_images_needed[subfolder] = per_folder_addition
    
#     # Distribute the remainder
#     for i in range(remainder):
#         subfolder_images_needed[subfolder_names[i]] += 1
    
#     return subfolder_images_needed

# # Function to balance images in each subfolder
# def balance_subfolder_images(base_path, main_folder, subfolder_name, additional_count):
#     subfolder_path = os.path.join(base_path, main_folder, subfolder_name)
    
#     # Check if the subfolder exists
#     if not os.path.exists(subfolder_path):
#         print(f"Subfolder {subfolder_name} does not exist in {os.path.join(base_path, main_folder)}. Skipping augmentation.")
#         return
    
#     image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
    
#     if len(image_files) == 0:
#         print(f"No images found in {subfolder_path}. Skipping augmentation.")
#         return
    
#     print(f"Augmenting {additional_count} images in {subfolder_path}")
#     for i in range(additional_count):
#         image_to_augment = random.choice(image_files)
#         image_path = os.path.join(subfolder_path, image_to_augment)
#         image = Image.open(image_path)
#         augmented_image = augment_image(image)
#         new_image_name = f"{os.path.splitext(image_to_augment)[0]}_aug_{i}{os.path.splitext(image_to_augment)[1]}"
#         augmented_image.save(os.path.join(subfolder_path, new_image_name))

# # Base directory paths containing all the subfolders
# base_directory = "/home/srikanth/Interns/RGB_images/Pinch"

# # Current number of images in each subfolder
# subfolder_counts = {
#     "comb": {
#         "comb_1": 605,
#         "comb_2": 605,
#         "comb_3": 605,
#         "comb_5": 605
#     },
#     "food_bag": {
#         "food_bag_1": 798,
#         "food_bag_2": 798,
#         "food_bag_3": 798,
#         "food_bag_4": 798,
#         "food_bag_5": 798,
#         "food_bag_6": 798,
#         "food_bag_7": 798,
#         "food_bag_8": 798
#     },
#     "glue_stick": {
#         "glue_stick_1": 801,
#         "glue_stick_2": 801,
#         "glue_stick_3": 801,
#         "glue_stick_4": 801,
#         "glue_stick_5": 801,
#         "glue_stick_6": 801
#     },
#     "keyboard": {
#         "keyboard_1": 709,
#         "keyboard_2": 709,
#         "keyboard_3": 709,
#         "keyboard_4": 709,
#         "keyboard_5": 709
#     },
#     "marker": {
#         "marker_1": 810,
#         "marker_2": 810,
#         "marker_3": 810,
#         "marker_4": 810,
#         "marker_5": 810,
#         "marker_6": 810,
#         "marker_7": 810,
#         "marker_8": 810,
#         "marker_9": 810
#     },
#     "plate": {
#         "plate_1": 712,
#         "plate_2": 712,
#         "plate_3": 712,
#         "plate_4": 712,
#         "plate_5": 712,
#         "plate_6": 712,
#         "plate_7": 712
#     },
#     "rubber_eraser": {
#         "rubber_eraser_1": 790,
#         "rubber_eraser_2": 790,
#         "rubber_eraser_3": 790,
#         "rubber_eraser_4": 790
#     },
#     "shampoo": {
#         "shampoo_1": 852,
#         "shampoo_2": 852,
#         "shampoo_3": 852,
#         "shampoo_4": 852,
#         "shampoo_5": 852,
#         "shampoo_6": 852
#     },
#     "toothbrush": {
#         "toothbrush_1": 605,
#         "toothbrush_2": 605,
#         "toothbrush_3": 605,
#         "toothbrush_4": 605,
#         "toothbrush_5": 605
#     }
# }
# total = 40726
# wanted = 75000
# # Total images to add
# total_images_to_add = total - wanted

# # Calculate additional images needed per subfolder
# additional_images_needed = calculate_additional_images({key: sum(value.values()) for key, value in subfolder_counts.items()}, total_images_to_add)

# # Balance images in each subfolder
# for subfolder, additional_count in additional_images_needed.items():
#     for sub_subfolder, count in subfolder_counts[subfolder].items():
#         balance_subfolder_images(base_directory, subfolder, sub_subfolder, additional_count // len(subfolder_counts[subfolder]))

# print("Image augmentation completed.")


import os
import random
from PIL import Image
from torchvision import transforms

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

# Calculate additional images required per subfolder
def calculate_additional_images(subfolder_counts, total_to_add):
    subfolder_names = list(subfolder_counts.keys())
    subfolder_images_needed = {subfolder: 0 for subfolder in subfolder_names}
    
    total_current_images = sum(subfolder_counts.values())
    total_subfolders = len(subfolder_names)
    
    # Distribute images as evenly as possible
    per_folder_addition = total_to_add // total_subfolders
    remainder = total_to_add % total_subfolders
    
    for subfolder in subfolder_names:
        subfolder_images_needed[subfolder] = per_folder_addition
    
    # Distribute the remainder
    for i in range(remainder):
        subfolder_images_needed[subfolder_names[i]] += 1
    
    return subfolder_images_needed

# Function to balance images in each subfolder
def balance_subfolder_images(base_path, main_folder, subfolder_name, additional_count):
    subfolder_path = os.path.join(base_path, main_folder, subfolder_name)
    
    # Check if the subfolder exists
    if not os.path.exists(subfolder_path):
        print(f"Subfolder {subfolder_name} does not exist in {os.path.join(base_path, main_folder)}. Skipping augmentation.")
        return
    
    image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
    
    if len(image_files) == 0:
        print(f"No images found in {subfolder_path}. Skipping augmentation.")
        return
    
    print(f"Augmenting {additional_count} images in {subfolder_path}")
    for i in range(additional_count):
        image_to_augment = random.choice(image_files)
        image_path = os.path.join(subfolder_path, image_to_augment)
        image = Image.open(image_path)
        augmented_image = augment_image(image)
        new_image_name = f"{os.path.splitext(image_to_augment)[0]}_aug_{i}{os.path.splitext(image_to_augment)[1]}"
        augmented_image.save(os.path.join(subfolder_path, new_image_name))

# Base directory paths containing all the subfolders
base_directory = "/home/srikanth/Interns/Hit-GPRec-merged/Palmar wrist neutral"

# Current number of images in each subfolder
subfolder_counts = {
    "01": 408,
    "02": 408,
    "03": 408,
    "04": 408,
    "05": 408,
    "06": 408,
    "07": 408,
    "08": 408,
    "09": 408,
    "10": 408,
    "11": 408,
    "12": 408,
    "13": 408,
    "14": 408,
    "15": 408,
    "16": 408,
    "17": 408,
    "18": 408,
    "19": 408,
    "20": 408,
    "21": 408,
    "22": 408,
    "23": 408,
    "24": 408,
    "25": 408,
    "26": 408,
    "27": 408,
    "28": 408,
    "29": 408,
    "30": 408,
    "31": 408,
    "32": 408,
    "33": 408,
    "34": 408,
    "35": 408
}

# Total images to add
#FOR RGB IMAGES
# total_images_to_add = 24000 - sum(sum(inner_dict.values()) for inner_dict in subfolder_counts.values())
#FOR HITGPREC
current_total_images = sum(subfolder_counts.values())
total_images_to_add = 24000 - current_total_images

# Calculate additional images needed per subfolder
additional_images_needed = calculate_additional_images({key: sum(value.values()) for key, value in subfolder_counts.items()}, total_images_to_add)

# Balance images in each subfolder
for subfolder, additional_count in additional_images_needed.items():
    for sub_subfolder, count in subfolder_counts[subfolder].items():
        balance_subfolder_images(base_directory, subfolder, sub_subfolder, additional_count // len(subfolder_counts[subfolder]))

print("Image augmentation completed.")
