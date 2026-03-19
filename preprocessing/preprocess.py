import os
from PIL import Image

# 🔥 THE FIX: Indha oru line thaan matter! Pillow-kitta size limit thevayilla nu solrom
Image.MAX_IMAGE_PIXELS = None 

# Namma folders irukka path
DATASET_PATH = 'dataset/train'
TARGET_SIZE = (224, 224)

def preprocess_images():
    print("[INFO] AI Preprocessing Start aagiduchu...")
    folders = ['cyclone', 'fire', 'flood', 'normal']
    total_processed = 0
    
    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.exists(folder_path):
            print(f"[WARNING] {folder} folder kedaikala!")
            continue
            
        images = os.listdir(folder_path)
        print(f"👉 Processing {len(images)} images in '{folder}' folder...")
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Image open pandrom
                img = Image.open(img_path)
                
                # Sila image 'RGBA' (transparent) illa Grayscale-la irukkum. Adha normal RGB-ku mathu
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # 224x224 ku resize pandrom
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # Original image idathulaye compress panni save pandrom
                img.save(img_path, 'JPEG', quality=85)
                total_processed += 1
                
            except Exception as e:
                # Ketta/Corrupted images irundha print pannitu delete pannidum
                print(f"[ERROR] Bad image {img_name}. Deleting it...")
                try:
                    os.remove(img_path) 
                except:
                    pass
                
    print(f"\n[SUCCESS] Mass! Total-a {total_processed} images compress aagiduchu. Ready for Training! 🔥")

if __name__ == "__main__":
    preprocess_images()