"""
This module defines the ImageUtils class, which contains various methods for reading images from a given folder path.
This can be done either sequentially if the number of images is small, or in parallel if the number of images is large.
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class ImageUtils:

    @staticmethod
    def read_single_image(args):
        """
        Reads a single image from a given path.
        :param image_path: The path to the image.
        :param dimensions: The dimensions to which the image should be resized.
        :return: The image.
        """

        image_path, dimensions = args
        image = Image.open(image_path)
        image = image.resize(dimensions)
        image = image.convert("RGB")            # Convert to RGB if the image has more than 3 channels
        return np.asarray(image)

    @staticmethod
    def read_sequential(folder_path, dimensions):
        """
        Reads all images from a given folder sequentially. Use if the number of images is small.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :return: A numpy array of images and a list of their filenames.
        """

        filenames = os.listdir(folder_path)
        images = []

        print(f"Reading {len(filenames)} images...")
        for filename in tqdm(filenames):
            image_path = os.path.join(folder_path, filename)
            image = ImageUtils.read_single_image((image_path, dimensions))
            images.append(image)
        
        return np.vstack(images), filenames

    @staticmethod
    def read_parallel(folder_path, dimensions):
        """
        Reads all images from a given folder in parallel. Use if the number of images is large.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :return: A numpy array of images and a list of their filenames.
        """

        filenames = os.listdir(folder_path)
        images = []

        print(f"Reading {len(filenames)} images...")
        num_processes = cpu_count()
        with Pool(num_processes) as pool, tqdm(total=len(filenames), desc="Loading Images") as pbar:
            args = [(os.path.join(folder_path, filename), dimensions) for filename in filenames]
            for image in pool.imap(ImageUtils.read_single_image, args):
                images.append(image)
                pbar.update(1)

        return np.vstack(images), filenames
    
    @staticmethod
    def read_images(folder_path, dimensions, threshold=1000):
        """
        Reads all images from a given folder, either sequentially or in parallel depending on whether or not the number of images exceeds a given threshold.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :param threshold: The threshold for the number of images above which to read in parallel.
        :return: A list of images and a list of their filenames.
        """

        if len(os.listdir(folder_path)) > threshold:
            return ImageUtils.read_parallel(folder_path, dimensions)
        else:
            return ImageUtils.read_sequential(folder_path, dimensions)
    
    @staticmethod
    def read_filenames(folder_path):
        """
        Reads all the filenames in a given folder.
        :param folder_path: The path to the folder containing the images.
        :return: A list of filenames.
        """

        return os.listdir(folder_path)