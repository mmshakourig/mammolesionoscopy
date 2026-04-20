import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np

# Keras / TensorFlow imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tqdm

class ShallowKerasCNN:
    def __init__(self, input_shape=(640, 640, 1), num_out=1):
        # Automatically select GPU if available
        self.device_name = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
        print(f"Initializing Keras model on {self.device_name}")
        
        with tf.device(self.device_name):
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                MaxPooling2D((2, 2)),
                
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(num_out, activation='sigmoid' if num_out == 1 else 'softmax')
            ])

    def compile(self, metrics):
            self.model.compile(optimizer='adam', loss='binary_crossentropy' if num_out == 1 else 'categorical_crossentropy', metrics=metrics)

    def train(self, x, y):
        # Add channel dimension if missing
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        with tf.device(self.device_name):
            # Compile and train the Keras model
            self.model.fit(x, y, epochs=10, batch_size=32)

    def predict(self, x):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        with tf.device(self.device_name):
            return self.model.predict(x)

    def get_model(self):
        return self.model

class ShallowCNN(nn.Module):
    """
    The PyTorch equivalent of the shallow Keras CNN.
    """
    def __init__(self, in_channels=1, num_classes=1):
        super(ShallowCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet50CNN(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True):
        super(ResNet50CNN, self).__init__()
        
        # We classify into Benign (0) vs Malignant (1)
        if use_pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet50()

        # Replace the final fully connected layer to output binary labels
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
