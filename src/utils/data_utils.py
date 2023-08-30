import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tqdm import tqdm


class DataGenerator(Sequence):
    def __init__(self, captions_dic, features_dic, tokenizer, batch_size, max_length):

        # make sure that the descriptions and features have the same keys
        assert set(captions_dic.keys()) == set(features_dic.keys()), "Captions and features don't have the same keys"

        # Store the provided data and parameters
        self.descriptions = captions_dic
        self.features = features_dic
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.vocab_size = len(tokenizer.word_index) + 1
        self.batch_size = batch_size
        self.keys = list(captions_dic.keys())

        
    def __len__(self):
        # Calculate the number of batches per epoch
        return len(self.keys) // self.batch_size

    def __getitem__(self, index):
        # Get the keys for the current batch
        batch_keys = self.keys[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x1, batch_x2, batch_y = [], [], []

        # Generate data for each key in the batch
        for key in batch_keys:
            description_list = self.descriptions[key]
            feature = self.features[key]
            inp_image, inp_seq, op_word = self.create_sequences(description_list, feature)
            batch_x1.extend(inp_image)
            batch_x2.extend(inp_seq)
            batch_y.extend(op_word)

        return [np.array(batch_x1), np.array(batch_x2)], np.array(batch_y)

    def create_sequences(self, desc_list, feature):
        x_1, x_2, y = [], [], []

        # Iterate over each description in the list
        for desc in desc_list:
            seq = self.tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

                x_1.append(feature)
                x_2.append(in_seq)
                y.append(out_seq)

        return x_1, x_2, y


class DataSplitter:

    def __init__(self, captions_dic, features_dic, tokenizer, batch_size):
        self.descriptions = captions_dic
        self.features = features_dic
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.keys = list(captions_dic.keys())
        self.max_length = self.get_max_desc_length()
    
    def split_data(self, val_split=0.2, shuffle=True):
        if val_split == 0:
            train_generator = DataGenerator(self.descriptions, self.features, self.tokenizer, self.batch_size, max_length=self.max_length)
            return train_generator, None
        
        # split the keys into train and validation sets
        train_keys, val_keys = self.split_keys(val_split, shuffle)

        # get the corresponding descriptions and features for the train and validation sets
        train_descriptions = {k: self.descriptions[k] for k in train_keys}
        train_features = {k: self.features[k] for k in train_keys}

        val_descriptions = {k: self.descriptions[k] for k in val_keys}
        val_features = {k: self.features[k] for k in val_keys}

        # create the generators
        train_generator = DataGenerator(train_descriptions, train_features, self.tokenizer, self.batch_size, max_length=self.max_length)
        val_generator = DataGenerator(val_descriptions, val_features, self.tokenizer, self.batch_size, max_length=self.max_length)

        return train_generator, val_generator


    def split_keys(self, val_split, shuffle=True):

        if val_split == 0:
            return self.keys, []

        # shuffle the keys
        if shuffle: np.random.shuffle(self.keys)

        # split the keys into train and validation sets
        split_index = int(len(self.keys) * (1 - val_split))
        train_keys, val_keys = self.keys[:split_index], self.keys[split_index:]

        return train_keys, val_keys
    
    def get_max_desc_length(self):
        """
        Get the maximum length of the captions in the captions dictionary.
        """
        max_length = 0
        for _, captions_list in tqdm(self.descriptions.items()):
            for caption in captions_list:
                max_length = max(max_length, len(caption.split()))
        
        return max_length