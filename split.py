import os
import shutil
import random
from collections import defaultdict

def split_dataset(images_folder, labels_folder, output_folder, split_ratio=0.8):
    # Group images by patient_id
    patient_images = defaultdict(list)
    for image_file in os.listdir(images_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_name = os.path.splitext(image_file)[0]
            patient_id = image_name.split('_')[0]  # Extract patient_id before the first underscore
            if os.path.exists(os.path.join(labels_folder, f"{image_name}.txt")):
                patient_images[patient_id].append(image_name)

    # Get list of patient_ids excluding '00' and shuffle
    patient_ids = list(patient_images.keys())
    if '00' in patient_ids:
        patient_ids.remove('00')
        val_patient_ids = ['00']  # Always assign patient '00' to validation set
    else:
        val_patient_ids = []

    # Shuffle and split patient_ids
    random.shuffle(patient_ids)
    split_index = int(len(patient_ids) * split_ratio)
    train_patient_ids = patient_ids[:split_index]
    val_patient_ids.extend(patient_ids[split_index:])

    # Copy images and labels to the corresponding directories
    for patient_id, images in patient_images.items():
        if patient_id in train_patient_ids:
            dest_folder = 'train'
        elif patient_id in val_patient_ids:
            dest_folder = 'val'
        else:
            continue  # Should not happen

        dest_image_folder = os.path.join(output_folder, dest_folder, 'images')
        dest_label_folder = os.path.join(output_folder, dest_folder, 'labels')
        os.makedirs(dest_image_folder, exist_ok=True)
        os.makedirs(dest_label_folder, exist_ok=True)

        for image_name in images:
            src_image_path = os.path.join(images_folder, f"{image_name}.jpg")
            src_label_path = os.path.join(labels_folder, f"{image_name}.txt")
            shutil.copy(src_image_path, os.path.join(dest_image_folder, f"{image_name}.jpg"))
            shutil.copy(src_label_path, os.path.join(dest_label_folder, f"{image_name}.txt"))

    print("Dataset split completed successfully with patient-wise organization.")

# Usage
images_folder = './six_class_images'
labels_folder = './six_class_labels'
output_folder = './output'
split_dataset(images_folder, labels_folder, output_folder)
