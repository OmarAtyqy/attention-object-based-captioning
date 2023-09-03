"""
This module implements the ImageUtils class, which is used to load and preprocess images.
"""

import concurrent.futures
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageUtils:

    @staticmethod
    def read_single_image(args):
        """
        Reads a single image from a given path.
        :param args: A tuple of (image_path, dimensions).
        :return: The image as a numpy array.
        """

        image_path, dimensions = args
        image = Image.open(image_path)
        image = image.resize(dimensions)

        # Convert to RGB if the image has more than 3 channels
        image = image.convert("RGB")

        # Convert to numpy array
        image = np.array(image)

        return image

    @staticmethod
    def read_sequential(folder_path, dimensions):
        """
        Reads all images from a given folder sequentially. Use if the number of images is small.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :param preprocess_function: The preprocess function to use.
        :return: a dictionary where the keys are the filenames and the values are the images.
        """

        filenames = os.listdir(folder_path)
        images_dic = {}

        print(f"Reading {len(filenames)} images sequentially...")
        for filename in tqdm(filenames):
            image_path = os.path.join(folder_path, filename)
            image = ImageUtils.read_single_image((image_path, dimensions))

            images_dic[filename] = image

        return images_dic

    @staticmethod
    def read_parallel(folder_path, dimensions):
        """
        Reads all images from a given folder in parallel while preserving the order. Use if the number of images is large.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :param preprocess_function: The preprocess function to use.
        :return: A dic where the keys are the filenames and the values are the images.
        """

        # Sort filenames to maintain order
        filenames = sorted(os.listdir(folder_path))
        images_dic = {}

        print(f"Reading {len(filenames)} images in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(ImageUtils.read_single_image, (os.path.join(
                folder_path, filename), dimensions)) for filename in filenames]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(filenames)):
                image = future.result()
                images_dic[filenames[futures.index(future)]] = image

        return images_dic

    @staticmethod
    def read_images(folder_path, dimensions, threshold=1000):
        """
        Reads all images from a given folder, either sequentially or in parallel depending on whether or not the number of images exceeds a given threshold.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :param threshold: The threshold for the number of images above which to read in parallel.
        :return: A dictionary where the keys are the filenames and the values are the images.
        """

        # check if the folderpath exists
        if not os.path.exists(folder_path):
            raise Exception(f"The folder {folder_path} does not exist.")

        if len(os.listdir(folder_path)) > threshold:
            images_dic = ImageUtils.read_parallel(folder_path, dimensions)
        else:
            images_dic = ImageUtils.read_sequential(folder_path, dimensions)

        return images_dic

    @staticmethod
    def preprocess_images(images_dic, preprocess_function):
        """
        Preprocesses the images.
        :param images_dic: A dictionary where the keys are the filenames and the values are the images.
        :param preprocess_function: The preprocess function to use.
        :return: A dictionary where the keys are the filenames and the values are the preprocessed images.
        """

        print("Preprocessing images...")
        for filename in tqdm(images_dic.keys()):
            images_dic[filename] = preprocess_function(images_dic[filename])

        return images_dic

    @staticmethod
    def get_importance_features(image, model):
        """
        Returns the importance features vector of an image.
        These features are calculated as follows:
        1. Run the image through the model.
        2. for each detected object, calculate the importance factor (Area * Confidence Score).
        3. rank the objects by importance and pick the first 292.
        4. Create a flattened vector of the following features for each picked object: X, Y, W, H, Confidence Score, Class, Importance Factor.
        5. Pad the vectorof length 2044 (292 * 7 = 2044) with 0s to a length of 2048.
        :param image: The image.
        :return: The importance features.
        """

        # get the model output
        results = model(image)

        # get the bounding boxes, classes and confidence scores
        X = results.xyxy[0][:, 0].cpu().numpy()
        Y = results.xyxy[0][:, 1].cpu().numpy()
        W = (results.xyxy[0][:, 2] - results.xyxy[0][:, 0]).cpu().numpy()
        H = (results.xyxy[0][:, 3] - results.xyxy[0][:, 1]).cpu().numpy()
        scores = results.xyxy[0][:, 4].cpu().numpy()
        classes = results.xyxy[0][:, 5].cpu().numpy()

        # calculate the importance factors
        importance_factors = W * H * scores

        # sort the importance factors in descending order and pick the first 292
        indices = np.argsort(importance_factors)[::-1][:292]

        # extract the 292 most important objects
        X = X[indices][:292]
        Y = Y[indices][:292]
        W = W[indices][:292]
        H = H[indices][:292]
        scores = scores[indices][:292]
        classes = classes[indices][:292]
        importance_factors = importance_factors[indices][:292]

        # create the importance features vector
        importance_features = np.concatenate(
            (X, Y, W, H, scores, classes, importance_factors))

        # pad the vector with 0s to a length of 2048
        importance_features = np.pad(
            importance_features, (0, 2048 - len(importance_features)), "constant")

        return importance_features

    @staticmethod
    def get_importance_features_dic(images_dic, model):
        """
        Returns the importance features vectors of a dictionary of images.
        :param images_dic: A dictionary where the keys are the filenames and the values are the images.
        :return: A dictionary where the keys are the filenames and the values are the importance features vectors.
        """

        print("Extracting importance features...")
        importance_features_dic = {}
        for filename in tqdm(images_dic.keys()):
            importance_features_dic[filename] = ImageUtils.get_importance_features(
                images_dic[filename], model)

        return importance_features_dic
