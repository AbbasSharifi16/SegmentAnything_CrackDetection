import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary

# Load your wall cracks dataset using OpenCV or Pillow
data_dir = 'SemanticSegmentationDefects'
image_dir = 'SemanticSegmentationDefects\ImageDatastore'
label_dir = 'SemanticSegmentationDefects\PixelLabelDatastore'

image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]

# Prepare your dataset by splitting it into training, validation, and testing sets
train_images, val_images, test_images = np.split(image_paths, [int(0.6*len(image_paths)), int(0.8*len(image_paths))])
train_labels, val_labels, test_labels = np.split(label_paths, [int(0.6*len(label_paths)), int(0.8*len(label_paths))])

# Define the number of output channels in the last layer of the SegmentAnything model
num_classes = 2 # wall cracks vs. background

# Load the pretrained SegmentAnything model weights from the .pth file
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu" #if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# image = cv2.imread('SemanticSegmentationDefects\ImageDatastore\IMG128.JPG')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# masks = mask_generator.generate(image)

# print(len(masks))
# print(masks[0].keys())
# segmentation = masks[0]['segmentation']
# Display the binary mask
# plt.imshow(binary_mask, cmap='gray')
# plt.show()

# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     polygons = []
#     color = []
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         img = np.ones((m.shape[0], m.shape[1], 3))
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:,:,i] = color_mask[i]
#         ax.imshow(np.dstack((img, m*0.35)))

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 

# Iterate over the image paths and generate masks for each image
for i, img_path in enumerate(image_paths):
    print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
    # Load the image using OpenCV
    image = cv2.imread(img_path)
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate the masks for the image
    masks = mask_generator.generate(image)

    # Save the masks as png files in a separate directory
    mask_dir = os.path.join(data_dir, "Masks")
    os.makedirs(mask_dir, exist_ok=True)
    mask_file = os.path.join(mask_dir, os.path.splitext(os.path.basename(img_path))[0] + ".png")
    plt.imsave(mask_file, masks[0]["segmentation"])
    
    
    
    

# # Define a colormap for the masks
# colors = np.array([
#     [0, 0, 0],         # background
#     [255, 0, 0],       # class 1 (red)
#     [0, 255, 0],       # class 2 (green)
#     [0, 0, 255],       # class 3 (blue)
#     [255, 255, 0],     # class 4 (yellow)
#     [255, 0, 255],     # class 5 (magenta)
#     [0, 255, 255],     # class 6 (cyan)
#     [255, 255, 255],   # class 7 (white)
# ])

# for i, img_path in enumerate(image_paths):
#     print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
#     # Load the image using OpenCV
#     image = cv2.imread(img_path)
#     # Convert the image to RGB format
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Generate the masks for the image
#     masks = mask_generator.generate(image)

#     # Create Boolean masks for each segmentation mask
#     boolean_masks = []
#     for i in range(1, 9):
#         segmentation = masks[i-1]['segmentation']
#         boolean_mask = np.where(segmentation, True, False)
#         boolean_masks.append(boolean_mask)

#     # Save the masks as png files in a separate directory
#     mask_dir = os.path.join(data_dir, "Masks")
#     os.makedirs(mask_dir, exist_ok=True)
#     for j, boolean_mask in enumerate(boolean_masks):
#         mask_file = os.path.join(mask_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_mask{j+1}.png")
#         plt.imsave(mask_file, boolean_mask.astype(np.uint8), cmap='gray')
        
        
        
# for i, img_path in enumerate(image_paths):
#     print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
#     # Load the image using OpenCV
#     image = cv2.imread(img_path)
#     # Convert the image to RGB format
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Generate the masks for the image
#     masks = mask_generator.generate(image)

#     # Create a figure with a grid of subplots
#     fig, axs = plt.subplots(2, 4, figsize=(12, 6))
#     fig.suptitle(f"Masks for Image {i+1}")

#     # Display each mask in a different subplot with a different color
#     colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
#     for j in range(8):
#         segmentation = masks[j]['segmentation']
#         boolean_mask = np.where(segmentation, True, False)
#         axs[j // 4, j % 4].imshow(boolean_mask.astype(np.uint8), cmap='gray')
#         axs[j // 4, j % 4].set_title(f"Mask {j+1}", color=colors[j])
#         axs[j // 4, j % 4].axis('off')

#     # Save the figure as a PNG file in a separate directory
#     mask_dir = os.path.join(data_dir, "Masks")
#     os.makedirs(mask_dir, exist_ok=True)
#     mask_file = os.path.join(mask_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_masks.png")
#     plt.savefig(mask_file)

import matplotlib.colors

for i, img_path in enumerate(image_paths):
    print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
    # Load the image using OpenCV
    image = cv2.imread(img_path)
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate the masks for the image
    masks = mask_generator.generate(image)

    # Combine the masks into a single RGB image with different colors for each mask
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    combined_mask = np.zeros_like(image)
    for j in range(8):
        segmentation = masks[j]['segmentation']
        color = np.array(matplotlib.colors.to_rgba(colors[j])).reshape(1, 1, 4)
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[segmentation] = np.array(color[0, 0, :-1] * 255, dtype=np.uint8)
        combined_mask += color_mask
        # Save the combined mask as a PNG file in a separate directory
    alpha = 0.5
    overlay = cv2.addWeighted(image, alpha, combined_mask, 1 - alpha, 0)
    mask_dir = os.path.join(data_dir, "Masks")
    os.makedirs(mask_dir, exist_ok=True)
    mask_file = os.path.join(mask_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_combined_mask.png")
    plt.imsave(mask_file, overlay)
    


    