"""
This module implements the CaptionHandler class, which contains various methods for reading captions from a given text file, as well
as methods to process, tokenize and pad the captions.
"""

import os
import re
import string

from tqdm import tqdm


class CaptionHandler:

    def __init__(self, filepath, filenames):
        """
        Initializes the CaptionHandler object.
        :param filepath: The path to the text file containing the captions.
        :param filenames: A list of filenames to read the captions for.
        """

        # check if the file exists
        if not os.path.exists(filepath):
            raise ValueError("The file {} does not exist.".format(filepath))

        self.filepath = filepath
        self.filenames = filenames
        self.captions_dic = self.read_captions()
    
    def read_captions_from_txt(self):
        """
        Reads the captions from the text file.
        :return: A dictionary where the keys are the filenames and the values are lists of captions.
        """

        with open(self.filepath, "r") as f:
            lines = f.readlines()

        # skip the header line and the get the number of lines
        lines = lines[1:]

        # create a dictionary to store the captions
        captions_dic = {k:[] for k in self.filenames}

        print("Reading captions...")
        for line in tqdm(lines):
            # split the line into filename and caption, use the first occurence of "," as the separator
            filename, caption = line.split(",", 1)

            # check if the filename is in the list of filenames
            if filename not in self.filenames:
                raise ValueError("Filename {} not found in the list of filenames.".format(filename))

            captions_dic[filename].append(self.process_text(caption))
        
        return captions_dic
    
    def read_captions_from_csv(self):
        """
        Reads the captions from the csv file.
        :return: A dictionary where the keys are the filenames and the values are lists of captions.
        """
        pass

    def read_captions(self):
        """
        Reads the captions from the filepath depending on the file extension.
        :return: A dictionary where the keys are the filenames and the values are lists of captions.
        """

        # check if the file is a text file or a csv file
        if self.filepath.endswith(".txt"):
            return self.read_captions_from_txt()
        elif self.filepath.endswith(".csv"):
            return self.read_captions_from_csv()
        else:
            raise ValueError("The file {} is not a text file or a csv file.".format(self.filepath))
    
    def process_text(self, text):
        """
        Processes a given text by converting it to lowercase, removing punctuation and numbers, and adding start and end tokens.
        :param text: The text to process.
        :return: The processed text.
        """

        # convert to lowercase
        text = text.lower()

        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # remove numbers
        text = re.sub(r"\d+", "", text)

        # remove extra whitespace
        text = text.strip()

        # add start and end tokens
        text = "<start> " + text + " <end>"

        return text