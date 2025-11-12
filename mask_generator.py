import cv2
import numpy as np
import os

# Paths
INPUT_DIR = r"I:\Sahana_3D_project\Dataset\images"
OUTPUT_DIR = r"I:\Sahana_3D_project\Dataset\masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def generate_masks(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Skipping invalid file: {image_path}")
        return

    gray = preprocess_image(img)

    # ==========================
    # 1️⃣ WALL DETECTION
    # ==========================
    # Walls are thick black lines → detect strong edges and dark pixels
    dark = cv2.inRange(img, (0, 0, 0), (80, 80, 80))
    walls = cv2.dilate(dark, np.ones((5, 5), np.uint8), iterations=2)

    # ==========================
    # 2️⃣ ROOM AREA DETECTION
    # ==========================
    # Invert walls to fill interior spaces (rooms)
    inv = cv2.bitwise_not(walls)
    flood = inv.copy()
    h, w = flood.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    rooms = cv2.bitwise_not(flood)

    # ==========================
    # 3️⃣ WINDOW DETECTION
    # ==========================
    # Windows → very thin long shapes (light gray or cyan colors)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_cyan = np.array([80, 30, 150])
    upper_cyan = np.array([110, 255, 255])
    windows = cv2.inRange(hsv, lower_cyan, upper_cyan)

    # Also include long thin rectangles in black/white plans
    thin_lines = cv2.Canny(gray, 80, 150)
    thin_lines = cv2.morphologyEx(thin_lines, cv2.MORPH_OPEN, np.ones((1, 15), np.uint8))
    windows = cv2.bitwise_or(windows, thin_lines)

    # ==========================
    # 4️⃣ DOOR DETECTION (optional)
    # ==========================
    # Doors usually appear as arcs or partial curves near walls
    edges = cv2.Canny(gray, 100, 200)
    doors = cv2.bitwise_and(edges, cv2.bitwise_not(walls))

    # ==========================
    # 5️⃣ FINAL COLOR MASK
    # ==========================
    color_mask = np.zeros_like(img)
    color_mask[:, :, 0] = walls      # Blue → Walls
    color_mask[:, :, 1] = rooms      # Green → Rooms
    color_mask[:, :, 2] = windows    # Red → Windows

    cv2.imwrite(save_path, color_mask)
    print(f"✅ Saved mask: {save_path}")

# ==========================
# 🚀 RUN FOR ALL IMAGES
# ==========================
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"mask_{filename}")
        generate_masks(input_path, output_path)

print("🎯 Mask generation complete! Check /masks folder.")
