from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import torch
from torchvision import datasets, transforms

# Define transformations
def preprocess_dataset_VIT(size = (224,224)):
    transform = transforms.Compose([
        transforms.Resize(size),  # Resize for ViT input
        transforms.ToTensor(),
    ])
    
    # Load Fashion MNIST dataset
    fashion_mnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    return fashion_mnist

def fashion_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)  # Shape: (60000, 28, 28, 1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train,y_train,x_test,y_test