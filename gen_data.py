import os
import cv2
import numpy as np

RAW_DIR = "raw_base_photos"
OUTPUT_DIR = "known_faces"

def aug_img(image, base_name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_orig.jpg"), image)

    #Brightness
    for b in [0.8, 1.2]:
        bright = cv2.convertScaleAbs(image, alpha=b, beta=0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_bright_{b}.jpg"), bright)

    #Saturations
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    for s in [0.6, 1.4]:
        hsv[:, :, 1] *= s
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        sat = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_sat_{s}.jpg"), sat)

    #GrayScales
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_gray.jpg"), gray_3ch)

def main():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        print(f"[INFO] Created '{RAW_DIR}' folder. Please place your expression photos here.")
        return
        
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print(f"[WARNING] No images found in '{RAW_DIR}'. Add your expression files first!")
        return
        
    for filename in files:
        path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(path)
        if img is not None:
            name_prefix = os.path.splitext(filename)[0]
            print(f"[PROCESSING] Augmenting expression set: {name_prefix}")
            aug_img(img, name_prefix)
            
    print("[SUCCESS] All expression sets successfully processed and expanded into 'known_faces'!")

if __name__ == "__main__":
    main()