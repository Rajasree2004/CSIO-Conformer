# import os

# def count_files_in_directory(directory, target_depth=3):
#     folder_counts = {}
#     directory = os.path.abspath(directory)
    
#     for root, dirs, files in os.walk(directory):
#         # Calculate the relative path and current depth
#         relative_path = os.path.relpath(root, directory)
#         depth = relative_path.count(os.sep) + 1
        
#         # Only process directories at the target depth
#         if depth == target_depth:
#             parent_folder = os.path.dirname(relative_path)
#             if parent_folder not in folder_counts:
#                 folder_counts[parent_folder] = 0
#             folder_counts[parent_folder] += len(files)
        
#         # Stop further traversal at the target depth
#         if depth >= target_depth:
#             dirs[:] = []
    
#     return folder_counts

# def print_folder_structure(folder_structure):
#     for folder, count in sorted(folder_structure.items()):
#         print(f"Folder: {folder}, Items: {count}")

# # Specify the root directory of the folder structure you want to read
# root_directory = '/home/srikanth/Dataset/RGB_images'


# # Get the folder structure with file counts up to the specified depth
# folder_structure = count_files_in_directory(root_directory)

# # Print the folder structure
# print_folder_structure(folder_structure)


import os

def count_files_in_directory(directory, target_depth=3):
    folder_counts = {}
    directory = os.path.abspath(directory)
    
    for root, dirs, files in os.walk(directory):
        # Calculate the relative path and current depth
        relative_path = os.path.relpath(root, directory)
        depth = relative_path.count(os.sep) + 1
        
        # Only process directories at the target depth
        if depth == target_depth:
            parent_folder = os.path.dirname(relative_path)
            if parent_folder not in folder_counts:
                folder_counts[parent_folder] = 0
            folder_counts[parent_folder] += len(files)
        
        # Stop further traversal at the target depth
        if depth >= target_depth:
            dirs[:] = []

    return folder_counts

def print_folder_structure(folder_structure):
    total_sum = 0
    for folder, count in sorted(folder_structure.items()):
        print(f"Folder:{folder},Items: {count}")
        total_sum += count
    print(f"Total Items: {total_sum}")

# Specify the root directory of the folder structure you want to read
root_directory = '/home/srikanth/Interns/RGB_images'

# Get the folder structure with file counts up to the specified depth
folder_structure = count_files_in_directory(root_directory)

# Print the folder structure and the total sum of items
print_folder_structure(folder_structure)

# /home/srikanth/Dataset/RGB_images
#     Palmar wrist neutral
#         coffee_mug
#             coffee_mug_1
#             coffee_mug_2
#             ...
#         food_can
#             food_can_1
#             food_can_2
#             ...
#     Palmar wrist pronated
#         food_cup
#             food_cup_1
#             food_cup_2
#             ...
#         food_jar
#             food_jar_1
#             food_jar_2
#             ...
#         kleenex
#             kleenex_1
#             kleenex_2
#             ...
import os

def count_files_in_directory(directory, target_depth=3):
    folder_counts = {}
    directory = os.path.abspath(directory)
    
    for root, dirs, files in os.walk(directory):
        # Calculate the relative path and current depth
        relative_path = os.path.relpath(root, directory)
        depth = relative_path.count(os.sep) + 1
        
        # Only process directories at the target depth
        if depth <= target_depth:
            parent_folder = os.path.dirname(relative_path)
            current_folder = os.path.basename(relative_path)
            
            if parent_folder not in folder_counts:
                folder_counts[parent_folder] = {}
            
            # Count files in current folder
            folder_counts[parent_folder][current_folder] = len(files)
        
    return folder_counts

def print_folder_structure(folder_structure):
    for main_folder, subfolders in sorted(folder_structure.items()):
        print(f"{main_folder}:")
        for subfolder, count in sorted(subfolders.items()):
            print(f"    {subfolder}: {count}")
            # Print sub-subfolders if they exist
            if isinstance(count, dict):
                for subsubfolder, subcount in sorted(count.items()):
                    print(f"        {subsubfolder}: {subcount}")

# Specify the root directory of the folder structure you want to read
root_directory = '/home/srikanth/Interns/RGB_images'

# Get the folder structure with file counts up to the specified depth
folder_structure = count_files_in_directory(root_directory)

# Print the folder structure with main folders, subfolders, and counts
print_folder_structure(folder_structure)


# import os
# from collections import defaultdict
# sum_of_each = 0

# def count_files_in_directory(directory, target_depth=2):
#     folder_counts = defaultdict(int)
#     directory = os.path.abspath(directory)
    
#     for root, dirs, files in os.walk(directory):
#         # Calculate the relative path and current depth
#         relative_path = os.path.relpath(root, directory)
#         depth = relative_path.count(os.sep) + 1
        
#         # Only process directories at the target depth
#         if depth == target_depth:
#             parent_folder = os.path.dirname(relative_path)
#             folder_counts[parent_folder] += len(files)
        
#         # Stop further traversal at the target depth
#         if depth >= target_depth:
#             dirs[:] = []

#     return folder_counts

# def print_folder_structure(folder_structure):
#     total_sum = 0
#     for folder, count in sorted(folder_structure.items()):
#         print(f"Folder: {folder}, Items: {count}")
#         total_sum += count
#     print("=" * 50)
#     print(f"Total Items: {total_sum}")

# def calculate_and_print_total_for_parent_folders(folder_structure, target_depth):
#     parent_folder_totals = defaultdict(int)
    
#     for folder, count in folder_structure.items():
#         # Get the parent folder name at the specified depth
#         parent_folder = '/'.join(folder.split('/')[:target_depth])
#         parent_folder_totals[parent_folder] += count
    
#     for parent_folder, total_count in parent_folder_totals.items():
#         print(f"{parent_folder} total: {total_count}")

# # Specify the root directory of the folder structure you want to read
# root_directory = '/home/srikanth/Interns/RGB_images'

# # Get the folder structure with file counts up to the specified depth
# folder_structure = count_files_in_directory(root_directory, target_depth=2)

# # Print the folder structure and the total sum of items
# print_folder_structure(folder_structure)

# # Calculate and print the total for each parent folder found at depth=2
# calculate_and_print_total_for_parent_folders(folder_structure, target_depth=2)
