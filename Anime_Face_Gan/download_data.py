import kagglehub
import os
import shutil

# Step 2: Download dataset using kagglehub
try:
    path = kagglehub.dataset_download("splcher/animefacedataset")
    print("Dataset downloaded successfully.")
    print("Path to dataset files:", path)
    
    # Step 3: Move the dataset to the target folder (if required)
    target_folder = r"C:\Users\cong_nguyen\Documents\Python\Start_Gan\Anime_Face_Gan\data"
    shutil.move(path, target_folder)
    print(f"Files moved to {target_folder}.")
except Exception as e:
    print("An error occurred while downloading or moving the dataset:", e)