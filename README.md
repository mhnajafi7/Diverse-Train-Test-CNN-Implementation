# Diverse-Train-Test-CNN-Implementation
In this notebook, we implement a CNN architecture for learning from a dataset with low similarity between test and train data (the [Dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083)). We use various techniques to improve predictions on the test data.
## Prequisties
First, we import the necessary libraries.
```python
import torch
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn # use nn functions like sigmoid, ReLu, softmax
import torchvision.transforms as transforms # use for transformerin on data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torchvision.models as models
import os
from PIL import Image
import io
import zipfile
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import time
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import json
from tqdm import tqdm
import io
! pip install gdown
```

## Dataset
We download the dataset JSON from Google Drive, which directly downloads from Kaggle. Using the "zipfile" library, we can work with the dataset without unzipping it. The train directory contains the training dataset, and the test directory contains the test dataset. We load both datasets using torch DataLoader.

## CNN Model
The `CNN` class defines a Convolutional Neural Network with the following architecture:

- **Convolutional Layers**:
  - `conv1`: 1 input channel, 128 output channels, 5x5 kernel, followed by batch normalization (`bn1`).
  - `pool1`: 5x5 average pooling with a stride of 2.
  - `conv2`: 128 input channels, 128 output channels, 3x3 kernel, followed by batch normalization (`bn2`).
  - `conv3`: 128 input channels, 128 output channels, 3x3 kernel, followed by batch normalization (`bn3`).
  - `conv4_1`: 128 input channels, 96 output channels, 3x3 kernel, followed by batch normalization (`bn4_1`).
  - `conv4_2`: 96 input channels, 96 output channels, 3x3 kernel, followed by batch normalization (`bn4_2`).
  - `conv5_1`: 96 input channels, 96 output channels, 3x3 kernel, followed by batch normalization (`bn5_1`).
  - `conv5_2`: 96 input channels, 96 output channels, 3x3 kernel, followed by batch normalization (`bn5_2`).
  - `pool2`: 5x5 average pooling with a stride of 2.

- **Dropout**:
  - Dropout layer with a probability of 0.5 after the convolutional layers and before the fully connected layers.

- **Fully Connected Layers**:
  - `fc1`: Fully connected layer with 96*54*54 input features and 128 output features, followed by a dropout (`dropout_fc1`) with a probability of 0.5.
  - `fc2`: Fully connected layer with 128 input features and `num_classes` output features.

- **Forward Pass**:
  - The forward method defines the sequence of operations: applying convolutional layers with ReLU activations, average pooling, dropout, flattening the output, and passing through the fully connected layers.

This architecture is designed to handle that dataset with low similarity between training and testing data, utilizing various techniques to improve prediction accuracy.

## Accuracy
The model accuracy after training for 10 epochs can reach 58 percent, which is a good accuracy for the 6-class classification problem. This is supported by the paper "Automated Defect Classification of Aluminium 5083 TIG Welding using HDR Camera and Neural Networks" by Daniel Bacioiu et al.. Also, the model's FPS (frames per second) during the test process is about 59 FPS.





