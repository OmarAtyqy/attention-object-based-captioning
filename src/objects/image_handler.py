"""
This module defines the ImageHandler class, which contains various methods for reading images from a given folder path.
This can be done either sequentially if the number of images is small, or in parallel if the number of images is large.
"""

import os
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageHandler:

    def __init__(self, folder_path, dimensions):
        """
        Initializes the ImageUtils object.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        """

        # check if the folder path exists
        if not os.path.exists(folder_path):
            raise ValueError("The folder path {} does not exist.".format(folder_path))

        self.folder_path = folder_path
        self.dimensions = dimensions

        self.images, self.filenames = self.read_images(self.folder_path, self.dimensions)
        self.images_dic = {k:v for k, v in zip(self.filenames, self.images)}

    def read_single_image(self, args):
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

    def read_sequential(self, folder_path, dimensions):
        """
        Reads all images from a given folder sequentially. Use if the number of images is small.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :return: An array of images and a list of their filenames.
        """

        filenames = os.listdir(folder_path)
        images = []

        print(f"Reading {len(filenames)} images...")
        for filename in tqdm(filenames):
            image_path = os.path.join(folder_path, filename)
            image = self.read_single_image((image_path, dimensions))
            images.append(image)
        
        return images, filenames

    def read_parallel(self, folder_path, dimensions):
        """
        Reads all images from a given folder in parallel. Use if the number of images is large.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :return: An array of images and a list of their filenames.
        """

        filenames = os.listdir(folder_path)
        images = []

        print(f"Reading {len(filenames)} images...")
        with Pool(cpu_count()) as p:
            for image in tqdm(p.imap(self.read_single_image, [(os.path.join(folder_path, filename), dimensions) for filename in filenames]), total=len(filenames)):
                images.append(image)
        
        return images, filenames

    
    def read_images(self, folder_path, dimensions, threshold=1000):
        """
        Reads all images from a given folder, either sequentially or in parallel depending on whether or not the number of images exceeds a given threshold.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :param threshold: The threshold for the number of images above which to read in parallel.
        :return: A list of images and a list of their filenames.
        """

        if len(os.listdir(folder_path)) > threshold:
            return self.read_parallel(folder_path, dimensions)
        else:
            return self.read_sequential(folder_path, dimensions)