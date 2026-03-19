import os

# Un dataset path
dataset_path = 'dataset/train'
total_images = 0

print("\n🔍 CHECKING DATASET IN FOLDERS 🔍")
print("=" * 35)

# Folders kulla poyi count pandrathu
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    
    # Adhu folder ah irundha mattum ulla poi count panu
    if os.path.isdir(folder_path):
        images_in_folder = len(os.listdir(folder_path))
        print(f"📁 {folder.upper().ljust(10)} : {images_in_folder} images")
        total_images += images_in_folder

print("=" * 35)
print(f"🔥 TOTAL IMAGES AVAILABLE: {total_images} 🔥")
print("=" * 35)

if total_images >= 12000:
    print("✅ Mass mamey! Target reached. AI ku semma virundhu irukku!")
else:
    print("⚠️ Mamey, count kammiya irukku. Folder ah check pannu.")