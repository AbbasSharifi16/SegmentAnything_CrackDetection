import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Load your wall cracks dataset using OpenCV or Pillow
data_dir = "SemanticSegmentationDefects"
image_dir = os.path.join(data_dir, "ImageDatastore")
label_dir = os.path.join(data_dir, "PixelLabelDatastore")

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

# Freeze the weights of all layers except for the last layer
for param in sam.parameters():
    param.requires_grad = False

in_channels = sam.cls_token.size(-1)
out_channels = num_classes

sam.fc.out_proj = nn.Linear(in_channels, out_channels)

# Compile the model with an appropriate loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sam.fc.parameters(), lr=1e-3)

# Train the model on your wall cracks dataset
num_epochs = 10

for epoch in range(num_epochs):
    sam.train()
    train_loss = 0
    for image_path, label_path in zip(train_images, train_labels):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label / 255 # Normalize the label to 0-1 range
        label = np.expand_dims(label, axis=-1) # Add a channel dimension to the label

        # Generate a mask for the image
        masks = mask_generator.generate(image)

        # Concatenate the image and mask along the channel dimension
        x = np.concatenate([image, masks], axis=-1)

        # Convert the input and target to PyTorch tensors and move to device
        x = torch.from_numpy(x.transpose((2, 0, 1))).float().to(device)
        y = torch.from_numpy(label.transpose((2, 0, 1))).float().to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x)

        # Compute the loss
        loss = criterion(outputs, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Print the average training loss for this epoch
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, train_loss / len(train_images)))

    # Save the model checkpoint
    torch.save(model.state_dict(), checkpoint_path)
