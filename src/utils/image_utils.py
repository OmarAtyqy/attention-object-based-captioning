import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_image(args):
    filename, folder_path, image_size, preprocess_input = args
    img_path = os.path.join(folder_path, filename)
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        if image_size:
            img = img.resize(image_size)  # Resize the image if needed
        img_array = np.array(img)
        if preprocess_input:
            img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image '{filename}': {e}")
        exit

def load_images_from_folder_parallel(folder_path, image_size=None, preprocess_input=None):
    filenames = os.listdir(folder_path)
    
    # Use multiprocessing to load images in parallel
    num_processes = cpu_count()
    with Pool(num_processes) as pool, tqdm(total=len(filenames), desc="Loading Images") as pbar:
        args = [(filename, folder_path, image_size, preprocess_input) for filename in filenames]
        images = []
        for img_array in pool.imap(load_image, args):
            images.append(img_array)
            pbar.update(1)
    
    return np.array(images)


def load_images_from_folder(folder_path, image_size=None, preprocess_input=None):
    """
    Load the images from a folder.
    """
    images = []
    
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Convert to RGB in case it has an alpha channel
            if image_size:
                img = img.resize(image_size)  # Resize the image if needed
            img_array = np.array(img)
            if preprocess_input:
                img_array = preprocess_input(img_array)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image '{filename}': {e}")
            exit
    
    return np.array(images)


def load_image_names(folder_path):
    """
    Load the names of the images in a folder.
    """
    filenames = []
    
    for filename in tqdm(os.listdir(folder_path)):
        filenames.append(filename)
    
    return filenames


def save_features(features, filename):
    """
    Save the features as a numpy array
    """
    # stack the features
    print("Stacking features...")
    stacked_features = np.vstack(features)
    print(f"Features shape: {stacked_features.shape}")

    # save the features
    print("Saving features...")
    np.save(filename, stacked_features)


def load_features(filename):
    """
    Load the features from a numpy array
    """
    return np.load(filename)
    


def load_features_as_dic(filename, filenames):
    """
    Load the features from a numpy array and return them as a dictionary where the keys are the filenames and the values are the features.
    """
    print("Loading features as a dictionary...")
    features = load_features(filename)
    return dict(zip(filenames, features))