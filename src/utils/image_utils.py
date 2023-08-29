import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle


def load_images_from_folder(folder_path, image_size=None, dtype=np.float16, preprocess_input=None):
    """
    Load images from a folder and returns them in a numpy array.
    folder_path: path to the folder containing the images.
    image_size: size of the images.
    dtype: type of the images. (The default is np.float16 to save memory)
    """
    images = []
    
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path)
            if image_size:
                img = img.resize(image_size)  # Resize the image if needed
            img_array = np.array(img, dtype=dtype)
            if preprocess_input:
                img_array = preprocess_input(img_array)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image '{filename}': {e}")
    
    return np.array(images)


def save_features(features, filename):
    """
    Save features to a file.
    features: numpy array of features.
    filename: name of the file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(features, f)