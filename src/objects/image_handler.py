"""
This module defines the ImageHandler class, which contains various methods for reading and processing images from a given folder path.
This can be done either sequentially if the number of images is small, or in parallel if the number of images is large.
ImageHandler also implements the calculation of the Importance Factor, which is used alongside the YOLOv5 model to bring the model's
attention to the most important parts of the image.
"""
# TODO: fix batch inference for YOLOv5

import concurrent.futures
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class ImageHandler:

    def __init__(self, folder_path, dimensions, confidence_threshold=0.2, preprocess_function=None):
        """
        Initializes the ImageUtils object.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :param batch_size: The batch size for the YOLOv5 inference.
        """

        # check if the folder path exists
        if not os.path.exists(folder_path):
            raise ValueError(
                "The folder path {} does not exist.".format(folder_path))

        # load the YOLOv5 model and set the confidence threshold to 0.4
        self.yolov5 = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", pretrained=True)
        self.yolov5.conf = confidence_threshold

        self.folder_path = folder_path
        self.dimensions = dimensions
        self.preprocess_function = preprocess_function

        self.images, self.filenames = self.read_images(
            self.folder_path, self.dimensions)
        self.images_dic = {k: v for k, v in zip(self.filenames, self.images)}

        # extract the features for each image
        self.features = self.extract_importance_factor_features(self.images)

        # build a dictionary mapping each filename to its features
        self.features_dic = {k: v for k, v in zip(
            self.filenames, self.features)}

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
        
        # Convert to RGB if the image has more than 3 channels
        image = image.convert("RGB")

        # preprocess the image if a preprocess function is provided
        if self.preprocess_function is not None:
            image = self.preprocess_function(image)

        return np.array(image)

    def read_sequential(self, folder_path, dimensions):
        """
        Reads all images from a given folder sequentially. Use if the number of images is small.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :return: An array of images and a list of their filenames.
        """

        filenames = os.listdir(folder_path)
        images = []

        print(f"Reading {len(filenames)} images sequentially...")
        for filename in tqdm(filenames):
            image_path = os.path.join(folder_path, filename)
            image = self.read_single_image((image_path, dimensions))
            images.append(image)

        return images, filenames

    def read_parallel(self, folder_path, dimensions):
        """
        Reads all images from a given folder in parallel while preserving the order. Use if the number of images is large.
        :param folder_path: The path to the folder containing the images.
        :param dimensions: The dimensions to which the images should be resized.
        :return: An array of images and a list of their filenames.
        """

        # Sort filenames to maintain order
        filenames = sorted(os.listdir(folder_path))
        images = []

        print(f"Reading {len(filenames)} images in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.read_single_image, (os.path.join(
                folder_path, filename), dimensions)) for filename in filenames]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(filenames)):
                image = future.result()
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
            images, filenames = self.read_parallel(folder_path, dimensions)
        else:
            images, filenames = self.read_sequential(folder_path, dimensions)

        return images, filenames

    def extract_importance_factor_features(self, images):
        """
        Extracts the features vector for each image, according to the Importance Factor. For each image, calculate the importance factor
        of the objects in the image, order them by importance, and then extract the 7 features (X, Y, W, H, score, class, importance) of the 292 most 
        important objects. Flatten the resulting array into a 2048-dimensional vector (7 * 292 = 2044 + 4 padding = 2048).
        The importance factor is calculated as follows: Area * Confidence score
        :param images: The images for which to calculate the Importance Factor. This should be a list of images.
        :param batch_size: The batch size for processing images in parallel.
        :return: an N * 2048 array of features, where N is the number of images.
        """

        all_features = []

        print("Extracting Importance Factor based features from images...")
        for image in tqdm(images):

            # infer
            results = self.yolov5(image)

            # get the bounding boxes and confidence scores
            X = results.xyxy[0][:, 0].cpu().numpy()
            Y = results.xyxy[0][:, 1].cpu().numpy()
            W = (results.xyxy[0][:, 2] - results.xyxy[0][:, 0]).cpu().numpy()
            H = (results.xyxy[0][:, 3] - results.xyxy[0][:, 1]).cpu().numpy()
            scores = results.xyxy[0][:, 4].cpu().numpy()
            classes = results.xyxy[0][:, 5].cpu().numpy()

            # calculate the importance factor
            importance = W * H * scores

            # sort the objects by importance
            indices = np.argsort(importance)[::-1]

            # extract the 292 most important objects
            X = X[indices][:292]
            Y = Y[indices][:292]
            W = W[indices][:292]
            H = H[indices][:292]
            scores = scores[indices][:292]
            classes = classes[indices][:292]
            importance = importance[indices][:292]

            # flatten the array into a 1D vector and pad it with 0s to make it 2048-dimensional
            features = np.concatenate(
                [X, Y, W, H, scores, classes, importance])
            features = np.pad(features, (0, 2048 - len(features)), "constant")

            all_features.append(features)

        return all_features
