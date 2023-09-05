# Image Captioning Model using Attention and Object Features

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Welcome to the GitHub repository for the replication of the paper "Image captioning model using attention and object features to mimic human image understanding" by Muhammad Abdelhadie Al-Malla, Assef Jafar, and Nada Ghneim. This project re-implements the work presented in the paper, exploring the fusion of attention mechanisms and object detection features to enhance the quality of image captions and simulate human-like image understanding.

The full paper can be found throughthe following [link](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00571-w).

## Abstract

This project is a replication and extension of the research paper "Image captioning model using attention and object features to mimic human image understanding." The paper presents a novel approach to image captioning that combines attention-based mechanisms with features derived from object detection. By harnessing these synergies, the model aims to generate captions that encapsulate more comprehensive image context and meaning, mirroring the way humans interpret images.

## Key Features

- Implementation of an attention-based Encoder-Decoder architecture.
- Integration of convolutional features from the ImageNet pre-trained Xception model.
- Incorporation of object features extracted from the YOLOv4 model pre-trained on MS COCO.
- Introduction of a novel positional encoding scheme named the "importance factor" to enrich object features.
- Enhancement of image caption quality through a combination of context-aware techniques.

## Usage

### Setting Up a Virtual Environment

1. **Create a Virtual Environment (Optional):** It's recommended to use a virtual environment to isolate the dependencies for this project. You can create one using the following commands (assuming you have Python and `virtualenv` installed):

```bash
# Create a virtual environment. This will create a virtual environment in the working directory
python -m venv env_name

# run the activation script
.\env_name\Scripts\activate
```

2. **Install Dependencies:** You can install the required Python libraries by running the following command while in your project directory:

```
python -m pip install -r requirements.txt
```

### Training

To train the model on your own dataset, simply open the file `train_model.py` and specify the following parameters:

```python
# ====================================== PARAMETERS ====================================== #

# path to the folder containing the images
images_folder_path = 'data/images'

# path to the file containing the captions
captions_path = 'data/captions.txt'

# preprocess function to use.
preprocess_function = tf.keras.applications.xception.preprocess_input

# validation split (percentage of the data used for validation)
val_split = 0

# batch size
# Make sure that your batch size is < the number of samples in both your training and validation datasets for the generators to work properly
batch_size = 32

# epochs
epochs = 10

# image dimensions
# The Xception model expects images of size 299x299, In order to change the input shape of the model inside the Encoder layer
# The dimensios should not be below 71
image_dimensions = (299, 299)

# embedding dimension (dimension of the Dense layer in the encoder and the Embedding layer in the decoder)
embedding_dim = 128

# number of units in the LSTM, Bahdanau attention and Dense layers
units = 256
```

before running the script using the command `python.exe train_script.py`

Make sure that your captions file is a `.txt` file that is structed as follows (don't forget to include the header line):

```plaintext
image,caption
filename1.jpg,caption 1.
filename1.jpg,caption 2.
filename2.jpg,caption 1.
...
```

## To-do

- [x] Implement data loading API.
- [x] Implement training script.
- [ ] Implement inference script.
- [ ] Implement batch inference for object detection instead of sequential inference.
- [ ] Implement `.csv` file support for captions reading (useful for reading the Flickr30k dataset).
