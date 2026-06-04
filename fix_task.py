import os
import shutil
import subprocess
import zipfile
from pathlib import Path

DATA_DIR = Path(r"D:\crop-disease-advisor\data")
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def download_and_extract(crop, dataset, expected_classes_min=None, expected_images_min=None):
    crop_dir = DATA_DIR / crop
    temp_dir = DATA_DIR / f"{crop}_temp"
    
    # 1. Delete everything inside
    print(f"Deleting {crop_dir}...")
    if crop_dir.exists():
        shutil.rmtree(crop_dir, ignore_errors=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    zip_name = dataset.split('/')[1] + ".zip"
    zip_path = DATA_DIR / zip_name
    if zip_path.exists():
        zip_path.unlink()
        
    # 2. Download
    print(f"Downloading {dataset}...")
    subprocess.run(f"kaggle datasets download -d {dataset} -p \"{DATA_DIR}\"", check=True, shell=True)
    
    # 3. Extract
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    print(f"Organizing {crop} dataset...")
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    class_folders = set()
    for root, dirs, files in os.walk(temp_dir):
        # check if this directory has images
        has_images = any(Path(f).suffix.lower() in VALID_EXTENSIONS for f in files)
        if has_images:
            # this is a class folder
            class_folders.add(root)
            
    # Move class folders to the target directory
    for folder in class_folders:
        folder_path = Path(folder)
        class_name = folder_path.name
        
        target_class_dir = crop_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy valid images
        for file in folder_path.iterdir():
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
                shutil.copy2(file, target_class_dir / file.name)
                
    # Cleanup temp dir and zip
    shutil.rmtree(temp_dir, ignore_errors=True)
    if zip_path.exists():
        zip_path.unlink()
    print(f"Finished setting up {crop}.\n")

if __name__ == "__main__":
    # PROBLEM 1 - Groundnut
    # The previous one 'muhammadazeemabbas/groundnut-leaves-dataset' had ~400 images for some classes.
    # We will use 'shirsmita/groundnut-leaf-disease-dataset' instead.
    download_and_extract("groundnut", "muhammadazeemabbas/groundnut-leaves-dataset")
    
    # PROBLEM 2 - Wheat
    download_and_extract("wheat", "kushagra3204/wheat-plant-diseases")
    
    # Verification
    print("Verification:")
    for crop in ['groundnut', 'wheat']:
        path = fr'D:\crop-disease-advisor\data\{crop}'
        print(f'\n{crop.upper()} classes:')
        if not os.path.exists(path):
            print("  Does not exist.")
            continue
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f'  {cls}: {count} images')
