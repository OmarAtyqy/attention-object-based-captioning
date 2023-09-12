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
    def get_importance_features(images_batch, model):
        """
        Returns the importance features vector of a batch of images.
        These features are calculated as follows:
        1. Run the images through the object detection model.
        2. for each image, for each detected object, calculate the importance factor (Area * Confidence Score).
        3. rank the objects by importance and pick the first 292.
        4. Create a flattened vector of the following features for each picked object: X, Y, W, H, Confidence Score, Class, Importance Factor.
        5. Pad the vectorof length 2044 (292 * 7 = 2044) with 0s to a length of 2048.
        :param image: The images.
        :return: A list of importance features vectors.
        """

        # run the images through the object detection model
        results = model(images_batch)

        # for each image, for each detected object, calculate the importance factor (Area * Confidence Score)
        importance_features_vectors_list = []

        # get the bounding boxes, classes and scores of the detected objects of all images
        for i in range(len(results.xyxy)):
            res = results.xyxy[i].cpu().numpy()

            # get the bounding boxes, classes and scores of the detected objects of the current image
            X = res[:, 0]
            Y = res[:, 1]
            W = res[:, 2] - res[:, 0]
            H = res[:, 3] - res[:, 1]
            classes = res[:, 5]
            scores = res[:, 4]

            # calculate the importance factors
            importance_factors = W * H * scores

            # sort the objects by importance
            sorted_indices = np.argsort(importance_factors)[::-1]

            # pick the first 292 objects
            sorted_indices = sorted_indices[:292]

            # create a flattened vector of the following features for each picked object: X, Y, W, H, Confidence Score, Class, Importance Factor
            importance_features_vector = np.concatenate(
                (X[sorted_indices], Y[sorted_indices], W[sorted_indices], H[sorted_indices], scores[sorted_indices], classes[sorted_indices], importance_factors[sorted_indices]))

            # pad the vector of length 2044 (292 * 7 = 2044) with 0s to a length of 2048
            importance_features_vector = np.pad(
                importance_features_vector, (0, 2048 - len(importance_features_vector)), 'constant')

            importance_features_vectors_list.append(importance_features_vector)

        return importance_features_vectors_list

    @staticmethod
    def get_importance_features_dic(images_dic, model, batch_size):
        """
        Returns the importance features vectors of a dictionary of images.
        :param images_dic: A dictionary where the keys are the filenames and the values are the images.
        :param model: The object detection model.
        :param batch_size: The batch size to use.
        :return: A dictionary where the keys are the filenames and the values are the importance features vectors.
        """

        importance_features_dic = {}

        # split the images into batches
        filenames = list(images_dic.keys())
        batches = [filenames[i:i + batch_size]
                   for i in range(0, len(filenames), batch_size)]

        # for each batch, get the importance features vectors
        print(
            f"Getting importance features in batches of {batch_size} images...")
        for batch in tqdm(batches):
            images_batch = [images_dic[filename] for filename in batch]
            importance_features_vectors_list = ImageUtils.get_importance_features(
                images_batch, model)

            for i in range(len(batch)):
                importance_features_dic[batch[i]
                                        ] = importance_features_vectors_list[i]

        return importance_features_dic
