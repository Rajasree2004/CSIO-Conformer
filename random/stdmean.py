# import os
# from PIL import Image
# import numpy as np

# # Define the root directory of your dataset
# root_dir = '/home/srikanth/Dataset/RGB_images'

# # Initialize an empty list to store image paths
# imgs_path = []

# # Traverse through subfolders to collect image paths
# for subdir, _, files in os.walk(root_dir):
#     for file in files:
#         # Check if the file is an image (you can add more image formats if needed)
#         if file.endswith(('.jpg', '.jpeg', '.png')):
#             # Construct the full path to the image
#             img_path = os.path.join(subdir, file)
#             # Append the image path to the list
#             imgs_path.append(img_path)

# # Calculate mean and standard deviation of pixel values
# rgb_values = np.concatenate(
#     [np.array(Image.open(img).getdata()) for img in imgs_path], 
#     axis=0
# ) / 255.

# # Calculate mean and standard deviation for each channel
# mu_rgb = np.mean(rgb_values, axis=0)
# std_rgb = np.std(rgb_values, axis=0)

# # Print the results
# print("Mean RGB values:", mu_rgb)
# print("Standard deviation RGB values:", std_rgb)

import os
from PIL import Image
import numpy as np

# Define the root directory of your dataset
root_dir = '/home/srikanth/Dataset/Hit-GPRec-merged'
num_images_per_folder = 500  # Number of images to use from each subfolder

# Initialize an empty list to store image paths
imgs_path = []

# Traverse through subfolders to collect image paths
for subdir, _, files in os.walk(root_dir):
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) > num_images_per_folder:
        image_files = image_files[:num_images_per_folder]
    for file in image_files:
        # Construct the full path to the image
        img_path = os.path.join(subdir, file)
        # Append the image path to the list
        imgs_path.append(img_path)

# Calculate mean and standard deviation of pixel values
rgb_values = np.concatenate(
    [np.array(Image.open(img).getdata()) for img in imgs_path], 
    axis=0
) / 255.

# Calculate mean and standard deviation for each channel
mu_rgb = np.mean(rgb_values, axis=0)
std_rgb = np.std(rgb_values, axis=0)

# Print the results
print("Mean RGB values:", mu_rgb)
print("Standard deviation RGB values:", std_rgb)

#for 200 images
#Mean RGB values: [0.51165725 0.47960767 0.46153016]
#Standard deviation RGB values: [0.22360833 0.22950601 0.24881599]
#for 300 images
# Mean RGB values: [0.51160183 0.4794664  0.46147034]
# Standard deviation RGB values: [0.22350493 0.22953787 0.24877995]


#hitgprec 500 images
#Mean RGB values: [0.74523671 0.75350752 0.73756239]
# Standard deviation RGB values: [0.23549667 0.23590666 0.25743267]