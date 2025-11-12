import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img.astype("float32") / 255.0

def load_and_preprocess_mask(mask_path, target_size=(256, 256)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    return (mask > 127).astype(np.uint8)  # Convert to binary mask

def evaluate_model():
    # Load the model
    model = load_model('model_output/unet_best.keras')
    
    # Setup paths
    test_images_dir = 'Dataset/images'
    test_masks_dir = 'Dataset/masks'
    
    # Get list of test images
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    accuracies = []
    precisions = []
    recalls = []
    
    # Create output directory for visualization
    os.makedirs('model_evaluation', exist_ok=True)
    
    print("Starting model evaluation...")
    
    for idx, img_name in enumerate(test_images[:5]):  # Test first 5 images for quick evaluation
        try:
            # Load and preprocess image
            img_path = os.path.join(test_images_dir, img_name)
            mask_path = os.path.join(test_masks_dir, img_name.replace('.jpg', '_mask.png'))
            
            if not os.path.exists(mask_path):
                print(f"Mask not found for {img_name}, skipping...")
                continue
                
            # Load image and true mask
            image = load_and_preprocess_image(img_path)
            true_mask = load_and_preprocess_mask(mask_path)
            
            # Get prediction
            pred_mask = model.predict(np.expand_dims(image, 0))[0]
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Calculate metrics
            accuracy = accuracy_score(true_mask.flatten(), pred_mask.flatten())
            precision = precision_score(true_mask.flatten(), pred_mask.flatten(), zero_division=1)
            recall = recall_score(true_mask.flatten(), pred_mask.flatten(), zero_division=1)
            
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            
            # Visualize results
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(true_mask, cmap='gray')
            plt.title('True Mask')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(pred_mask, cmap='gray')
            plt.title(f'Predicted Mask\nAcc: {accuracy:.2f}')
            plt.axis('off')
            
            plt.savefig(f'model_evaluation/evaluation_{idx}.png')
            plt.close()
            
            print(f"Processed {img_name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue
    
    # Calculate average metrics
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    print("\nOverall Model Performance:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

if __name__ == "__main__":
    evaluate_model()
