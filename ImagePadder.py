import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import concurrent.futures

# Directory where all the subfolders and images are stored
root_dir = 'AtlasDermatigo_Augmented'
output_dir = 'AtlasDermatigo_Augmented_Preprocessed'  # Specify output directory to save padded images

# Mean and standard deviation for padding pixel values (given)
mean = [0.6253888010978699, 0.4681059718132019, 0.40517428517341614]
std = [0.22259929776191711, 0.19853103160858154, 0.20957979559898376]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to pad an image
def pad_image_to_440(image):
    # Convert PIL image to NumPy array
    np_img = np.array(image)

    # Get current image dimensions
    height, width, _ = np_img.shape

    # Calculate padding
    pad_w = (440 - width) // 2 if width < 440 else 0
    pad_h = (440 - height) // 2 if height < 440 else 0

    # Create a blank canvas of 440x440
    padded_image = np.zeros((440, 440, 3), dtype=np.uint8)

    # Fill the padding area with random colors sampled from the distribution
    # Sample padding colors once for the entire image
    padding_colors = np.random.normal(mean, std, (440, 440, 3)).clip(0, 1) * 255

    # Center the original image on the padded canvas
    if pad_h > 0 or pad_w > 0:
        padded_image[pad_h:pad_h + height, pad_w:pad_w + width] = np_img
        # Fill padding area
        padded_image[:pad_h, :] = padding_colors[:pad_h, :]
        padded_image[pad_h + height:, :] = padding_colors[pad_h + height:, :]
        padded_image[:, :pad_w] = padding_colors[:, :pad_w]
        padded_image[:, pad_w + width:] = padding_colors[:, pad_w + width:]
    else:
        padded_image = np_img

    # Convert back to PIL Image
    return Image.fromarray(padded_image)

def process_image(args):
    folder_name, image_name, image_path = args
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Ensure image is in RGB format

            # Pad image to 440x440 if needed
            padded_img = pad_image_to_440(img)

            # Save padded image to output directory
            output_image_path = os.path.join(output_dir, folder_name)
            os.makedirs(output_image_path, exist_ok=True)

            padded_img.save(os.path.join(output_image_path, image_name))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    # Get list of all folders and images in the root directory
    all_images = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                all_images.append((folder_name, image_name, image_path))

    # Process all images using multiprocessing with tqdm for progress tracking
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, all_images), total=len(all_images), desc="Processing images", ncols=80))
