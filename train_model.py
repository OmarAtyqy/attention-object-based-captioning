"""
This script is used to train the model.
"""

from src.scripts.load_data import load_data


if __name__ == '__main__':
    
    # path to the folder containing the images
    images_folder_path = 'data/test_data/test'

    # path to the file containing the captions
    captions_path = 'data/test_data/test.txt'

    # load the data and unpack it
    data_dic = load_data(images_folder_path, captions_path)
    images_dic = data_dic['images_dic']
    captions_dic = data_dic['captions_dic']
    importance_features_dic = data_dic['importance_features_dic']
    tokenizer = data_dic['tokenizer']
    max_caption_length = data_dic['max_caption_length']