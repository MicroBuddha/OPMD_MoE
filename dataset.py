import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform=None):
        self.samples = []
        self.transform = transform
        self.images_folder = images_folder
        self.labels_folder = labels_folder

        # Collect all samples
        for image_file in os.listdir(images_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_name = os.path.splitext(image_file)[0]
                image_path = os.path.join(images_folder, image_file)
                label_path = os.path.join(labels_folder, f"{image_name}.txt")

                if not os.path.exists(label_path):
                    continue  # Skip if label file does not exist

                # Read label file
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines

                    # Parse class ID and bounding box coordinates
                    parts = line.split()
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]

                    # Store the sample as a tuple
                    self.samples.append({
                        'image_path': image_path,
                        'class_id': class_id,
                        'bbox': bbox
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        class_id = sample['class_id']
        bbox = sample['bbox']  # Normalized coordinates
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        polygon = [(bbox[i] * width, bbox[i + 1] * height) for i in range(0, len(bbox), 2)]
        # Convert normalized bbox to pixel coordinates
        x_min = min([point[0] for point in polygon])
        y_min = min([point[1] for point in polygon])
        x_max = max([point[0] for point in polygon])
        y_max = max([point[1] for point in polygon])

        # # Calculate corner points
        # x_min = max(int(x_center - box_width / 2), 0)
        # y_min = max(int(y_center - box_height / 2), 0)
        # x_max = min(int(x_center + box_width / 2), width)
        # y_max = min(int(y_center + box_height / 2), height)
        # Create mask
        if len(bbox) == 4:
            # If the label contains only 4 elements, treat it as a bounding box (x_center, y_center, width, height)
            # x_min, y_min, x_max, y_max = bbox[:4]
            # x_min *= width
            # y_min *= height
            # x_max *= width
            # y_max *= height

            # Calculate corner points for rectangle
            # x_min = max(int(x_center - box_width / 2), 0)
            # y_min = max(int(y_center - box_height / 2), 0)
            # x_max = min(int(x_center + box_width / 2), width)
            # y_max = min(int(y_center + box_width / 2), height)

            # Create a polygon mask for the bounding box
            polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        else:
            # Otherwise, treat it as a polygon with multiple points (normalized x1, y1, x2, y2, ..., xn, yn)
            polygon = [(bbox[i] * width, bbox[i + 1] * height) for i in range(0, len(bbox), 2)]

        mask = Image.new('L', (width, height), 0)
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)

        # Apply mask to the image
        mask = np.array(mask)
        image_array = np.array(image)
        masked_image = image_array * mask[:, :, np.newaxis]  # Apply mask to each channel

        # Convert back to PIL Image
        masked_image = Image.fromarray(masked_image)
        cropped_image = masked_image.crop((x_min, y_min, x_max, y_max))

        # Apply transformations if any
        if self.transform:
            cropped_image = self.transform(cropped_image)
            # masked_image = self.transform(masked_image)

        # Convert class_id to tensor
        class_id = torch.tensor(class_id, dtype=torch.long)

        return cropped_image, class_id
        
def visualize_batch(images, labels, classes=None, batch_size=32):
    """
    Visualizes a batch of images and their corresponding labels.

    Args:
    - images (torch.Tensor): A batch of images.
    - labels (torch.Tensor): Corresponding labels for the images.
    - classes (list): List of class names corresponding to the class indices.
    - batch_size (int): Number of images to display from the batch.

    """
    # Convert images from Tensor to NumPy format for visualization
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Rearrange dimensions to (batch_size, height, width, channels)
    
    # Unnormalize images if they were normalized
    images = images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Reversing the normalization
    images = np.clip(images, 0, 1)  # Clip values to [0, 1] range for visualization

    # Plot the batch
    plt.figure(figsize=(15, 8))
    for i in range(min(batch_size, images.shape[0])):
        plt.subplot(4, 8, i+1)  # Adjust the grid size as needed
        plt.imshow(images[i])
        label = labels[i].item()
        if classes:
            plt.title(f'Class: {classes[label]}')
        else:
            plt.title(f'Class ID: {label}')
        plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig("vis.png")
if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset instances with the corrected class name
    train_dataset = CustomImageDataset(
        images_folder='/home/luffy/data/data_opmd/OPMD_new/output/train/images',
        labels_folder='/home/luffy/data/data_opmd/OPMD_new/output/train/labels',
        transform=transform
    )

    val_dataset = CustomImageDataset(
        images_folder='/home/luffy/data/data_opmd/OPMD_new/output/val/images',
        labels_folder='/home/luffy/data/data_opmd/OPMD_new/output/val/labels',
        transform=transform
    )

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Example of loading a data sample
    # for images, labels in train_loader:
    #     print(f'Images batch shape: {images.size()}')  # Shape of the image tensor batch
    #     print(f'Labels batch shape: {labels.size()}')  # Shape of the label tensor batch
    #     break
    class_names = [f'class{i}' for i in range(6)]  # Adjust this list as per your dataset

    # Visualize a batch from the train_loader
    for images, labels in train_loader:
        visualize_batch(images, labels, classes=class_names)
        break