from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Convert image to numpy array
        img = np.array(img)
        # Add Gaussian noise
        noise = np.random.normal(self.mean, self.std, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255)  # Ensure pixel values are in [0, 255]
        return Image.fromarray(noisy_img.astype(np.uint8))

# Define individual transformations
horizontal_flip = transforms.RandomHorizontalFlip()
vertical_flip = transforms.RandomVerticalFlip()
rotation = transforms.RandomRotation(30)
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
resized_crop = transforms.RandomResizedCrop(224)
#gaussian_noise = GaussianNoise(mean=0, std=25)  # Adjust std as needed

def apply_probabilistic_transform(image):
    # Define the probability for each transformation
    transform_probs = {
        horizontal_flip: 0.5,
        vertical_flip: 0.5,
        rotation: 0.5,
        color_jitter: 0.5,
        resized_crop: 0.5,
        #gaussian_noise: 0.5
    }
    
    # Apply transformations based on their probabilities
    if random.random() < transform_probs[horizontal_flip]:
        image = horizontal_flip(image)
    if random.random() < transform_probs[vertical_flip]:
        image = vertical_flip(image)
    if random.random() < transform_probs[rotation]:
        image = rotation(image)
    if random.random() < transform_probs[color_jitter]:
        image = color_jitter(image)
    if random.random() < transform_probs[resized_crop]:
        image = resized_crop(image)
    #if random.random() < transform_probs[gaussian_noise]:
    #   image = gaussian_noise(image)
    
    return image

def is_image_duplicate(image_path, existing_images):
    return os.path.basename(image_path) in existing_images

# Directory paths
input_dir = 'AtlasDermatigo'
output_dir = 'AtlasDermatigo_Augmented'
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Apply probabilistic augmentations and save images
for label_folder in os.listdir(input_dir):
    label_folder_path = os.path.join(input_dir, label_folder)
    if os.path.isdir(label_folder_path):
        print(f"Processing folder: {label_folder}")  # Debug statement
        # Ensure label subdirectory exists in the output directory
        output_label_folder_path = os.path.join(output_dir, label_folder)
        os.makedirs(output_label_folder_path, exist_ok=True)
        
        # Keep track of existing images to avoid duplicates
        existing_images = set(os.listdir(output_label_folder_path))
        
        # Save the original image
        for filename in os.listdir(label_folder_path):
            img_path = os.path.join(label_folder_path, filename)
            print(f"Processing image: {filename}")  # Debug statement
            image = Image.open(img_path)
            original_image_path = os.path.join(output_label_folder_path, filename)
            image.save(original_image_path)  # Save the original image
            
            # Add the original image to the set of existing images
            existing_images.add(filename)

            # Apply transformations multiple times with different combinations
            for i in range(5):  # Create 5 augmented versions of each image
                augmented_image = apply_probabilistic_transform(image)
                augmented_filename = f'{os.path.splitext(filename)[0]}_aug_{i}.jpeg'
                augmented_image_path = os.path.join(output_label_folder_path, augmented_filename)
                
                if not is_image_duplicate(augmented_filename, existing_images):
                    augmented_image.save(augmented_image_path)
                    existing_images.add(augmented_filename)
                else:
                    print(f"Duplicate found and skipped: {augmented_filename}")  # Debug statement