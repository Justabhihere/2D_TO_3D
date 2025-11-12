from flask import Flask, render_template, request, redirect, url_for, abort, jsonify
import os
import uuid
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from mask_to_3d_unified import mask_to_3d_and_save
import traceback

def generate_mask(file_path, uid):
    """
    Generate a mask from the uploaded image using advanced wall detection.
    This detects walls as black/dark lines and creates proper architectural masks.
    
    Args:
        file_path: Path to the input image
        uid: Unique identifier for the output file
        
    Returns:
        Path to the generated mask
    """
    try:
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image from {file_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # ==========================
        # [1] WALL DETECTION
        # ==========================
        # Walls are thick black lines - detect strong edges and dark pixels
        dark = cv2.inRange(image, (0, 0, 0), (80, 80, 80))
        walls = cv2.dilate(dark, np.ones((5, 5), np.uint8), iterations=2)
        
        # ==========================
        # [2] ROOM AREA DETECTION
        # ==========================
        # Invert walls to fill interior spaces (rooms)
        inv = cv2.bitwise_not(walls)
        flood = inv.copy()
        h, w = flood.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        rooms = cv2.bitwise_not(flood)
        
        # ==========================
        # [3] WINDOW DETECTION
        # ==========================
        # Windows - very thin long shapes (light gray or cyan colors)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_cyan = np.array([80, 30, 150])
        upper_cyan = np.array([110, 255, 255])
        windows = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Also include long thin rectangles in black/white plans
        thin_lines = cv2.Canny(gray, 80, 150)
        thin_lines = cv2.morphologyEx(thin_lines, cv2.MORPH_OPEN, np.ones((1, 15), np.uint8))
        windows = cv2.bitwise_or(windows, thin_lines)
        
        # ==========================
        # [4] FINAL COLOR MASK
        # ==========================
        color_mask = np.zeros_like(image)
        color_mask[:, :, 0] = walls      # Blue - Walls
        color_mask[:, :, 1] = rooms      # Green - Rooms  
        color_mask[:, :, 2] = windows    # Red - Windows
        
        # Save the mask
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_mask.png")
        cv2.imwrite(mask_path, color_mask)
        
        print(f"[OK] Mask generated with {cv2.countNonZero(walls)} wall pixels, {cv2.countNonZero(rooms)} room pixels")
        return mask_path
        
    except Exception as e:
        print(f"Error generating mask: {e}")
        # Fallback: return original image path
        return file_path

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

@app.errorhandler(400)
def bad_request(error):
    return render_template("error.html", error_message=error.description), 400


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("error.html", error_message="Internal Server Error"), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle file upload and process with 3D generation.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Generate unique ID for this conversion
        uid = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{filename}")
        file.save(file_path)
        
        # Process with Tensor.Art model first
        print("Processing with Tensor.Art model...")
        try:
            try:
                from tensor_art_model import process_image_with_tensor_art
            except ImportError:
                from tensor_art_model_simple import process_image_with_tensor_art
            tensor_features = process_image_with_tensor_art(file_path)
            print(f"Tensor.Art features extracted: {list(tensor_features.keys()) if tensor_features else 'None'}")
        except Exception as e:
            print(f"Tensor.Art processing failed: {e}")
            tensor_features = None
        
        # Generate mask from uploaded image
        print("Generating mask...")
        mask_path = generate_mask(file_path, uid)
        
        # Always generate 3D model with the enhanced mask_to_3d function
        print("Converting mask to 3D model with enhanced features...")
        obj_path = mask_to_3d_and_save(mask_path, uid, "static/model_output")
        
        # Respond with core outputs only (Gemini integration removed)
        return jsonify({
            'uid': uid,
            'model_url': f'/static/model_output/{uid}_3d_model.obj',
            'texture_url': f'/static/model_output/{uid}_texture.jpg',
            'enhanced_features': True
        })
        
    except Exception as e:
        print(f"Upload processing error: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/view/<model_name>')
def view_3d(model_name):
    return render_template('view3d.html', model_name=model_name)


@app.route('/walkthrough/<model_name>')
def walkthrough_3d(model_name):
    return render_template('walkthrough.html', model_name=model_name)


if __name__ == '__main__':
    # Run without debug mode to prevent constant reloading
    app.run(debug=False, host='127.0.0.1', port=5000)
