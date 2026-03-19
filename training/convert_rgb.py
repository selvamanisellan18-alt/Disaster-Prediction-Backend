import os
import cv2

# 1. USE ABSOLUTE PATHS FOR BOTH (Remember the 'r' before the quotes!)
input_dir = r"E:\Disaster_Backend (1)\Disaster_Backend\dataset2\sen12flood\sen12floods_s2_source\sen12floods_s2_source" 
output_dir = r"output_pngs" 

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# 2. Print the exact absolute path so you know exactly where to look
print(f"Saving images EXACTLY to: {os.path.abspath(output_dir)}")
print("-" * 40)
m=0
for filename in os.listdir(input_dir):
    print(filename)
    
    
        
    print('hello')
    m+=1
    b2_path = os.path.join(input_dir,filename, "B02.tif")
    b3_path = os.path.join(input_dir,filename, "B03.tif")
    b4_path = os.path.join(input_dir,filename, "B04.tif")

    # Read the 16-bit images
    print("Reading bands...")
    b2 = cv2.imread(b2_path, cv2.IMREAD_UNCHANGED)
    b3 = cv2.imread(b3_path, cv2.IMREAD_UNCHANGED)
    b4 = cv2.imread(b4_path, cv2.IMREAD_UNCHANGED)

    # --- 3. MERGE AND NORMALIZE ---
    if b2 is not None and b3 is not None and b4 is not None:
        # Stack them in B-G-R order for OpenCV
        print("Stacking bands into a single image...")
        bgr_image = cv2.merge((b2, b3, b4))
        
        # Normalize the 16-bit satellite data to 8-bit (0-255)
        # We normalize the whole stacked image together to keep the color balanced
        bgr_normalized = cv2.normalize(bgr_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Optional brightness boost: Satellite images can sometimes look a bit dark 
        # when linearly normalized due to bright clouds skewing the math.
        # bgr_normalized = cv2.convertScaleAbs(bgr_normalized, alpha=1.5, beta=0) 
        
        # Save the final color image
        x=str(m)+'.png'
        success = cv2.imwrite(os.path.join(output_dir,x), bgr_normalized)
        
        if success:
            print(f"Success! Saved true-color image to:\n{os.path.join(output_dir,x)}")
        else:
            print("FAILED to save. Check folder permissions.")
    else:
        print("Error: Could not find one or more of the required band files (B02, B03, B04).")
        print("Make sure your input_dir is correct and the files exist.")

print("-" * 40)
print("Processing finished!")