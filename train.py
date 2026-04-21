from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    device = 0  # Use GPU 0

else:
    print("No GPU available, using CPU")
    device = "cpu"

# Load a segmentation model (yolo11m-seg.pt)
# model = YOLO("./models/yolo11m-seg.pt") 
model = YOLO("./runs/segment/train6/weights/epoch70.pt") 

# Train the model on the Carparts Segmentation dataset
# print("Starting training...")
# results = model.train(
#     data="./datasets/carparts-seg/carparts-seg.yaml", 
#     epochs=100, 
#     imgsz=640,
#     device=device,
#     batch=32,  # Increase batch size for GPU
#     workers=8,  # Increase workers for faster data loading
#     verbose=True,  # Enable verbose output
#     plots=True,  # Generate plots
#     save=True,  # Save checkpoints
#     save_period=10  # Save every 10 epochs
# )

# After training, validate the model's performance on the validation set
# print("Starting validation...")
# val_results = model.val(
#     data="./datasets/carparts-seg/carparts-seg.yaml",
#     device=device,
#     verbose=True
# )

# Print validation results
# print("Validation results:")
# print(f"mAP50: {val_results.box.map50:.4f}")
# print(f"mAP50-95: {val_results.box.map:.4f}")
# if hasattr(val_results, 'mask'):
#     print(f"Segmentation mAP50: {val_results.mask.map50:.4f}")
#     print(f"Segmentation mAP50-95: {val_results.mask.map:.4f}")

# # Perform prediction on new images with segmentation
# print("Starting prediction...")
results = model.predict(
    "./test/img/su7.png", 
    save=True, 
    project="./test", 
    name="results", 
    exist_ok=True, 
    conf=0.25,
    device=device
)

# Create results directory if it doesn't exist
os.makedirs("./test/results", exist_ok=True)

for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    
    result.save(filename="./test/results/last_su7_pred.png")
    
    # Print segmentation information
    print("Processing result:")
    print(f"Number of segmentation masks: {len(xy)}")
    if len(xy) > 0:
        print(f"First mask polygon points: {len(xy[0])}")
        print(f"Masks shape: {masks.shape}")
    
    # Visualize the segmentation masks
    if len(xy) > 0:
        # Load the original image
        img = cv2.imread("./test/img/su7.png")
        
        # Draw each segmentation mask
        for i, polygon in enumerate(xy):
            # Convert polygon points to integers
            polygon = polygon.astype(int)
            
            # Draw the polygon on the image
            cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Fill the polygon with semi-transparent color
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon], color=(0, 255, 0, 100))
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        
        # Save the visualized image
        cv2.imwrite("./test/results/last_su7_seg_visualized.png", img)
        print("Segmentation visualization saved to ./test/results/last_su7_seg_visualized.png")

print("Training and prediction completed successfully!")

