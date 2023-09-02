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
