# PixelCNN-implementation

# Description:
PixelCNN is an autoregressive generative model that generates images by modeling the conditional distribution of each pixel given the pixels above and to the left. It is known for generating high-quality images pixel by pixel, which can be used for image generation and completion tasks.

In this project, we’ll implement a PixelCNN for generating images one pixel at a time based on the CIFAR-10 dataset.

# ✅ What It Does:
* Defines a PixelCNN model that generates images pixel by pixel, conditioned on the previously generated pixels

* Trains on the CIFAR-10 dataset, learning to generate images based on pixel-level dependencies

* Uses cross-entropy loss for training the network and Adam optimizer for updating weights

* Generates 32x32 RGB images from random noise as input

