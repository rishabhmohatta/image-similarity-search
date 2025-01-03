from models.getresnet50model import getResNet50Model
from models.autoencodermodel import AutoencoderModel
from models.siamese_model import SiameseTripletModel
from models.VisionTransformermodel import ViT
from utils.extract_feature import extract_resnet_feature,extract_autoencoder_feature,extract_siamese_feature,extract_vit_feature
from utils.indexing_faiss import vector_database,search_database
from utils.Visualize import plot_results,plot_results_save
from utils.dataset import fashion_dataset,preprocess_dataset_VIT
from sklearn.model_selection import train_test_split
from settings.constants import AUTOENCODER_MODEL_PATH,SIAMESE_MODEL_PATH
import numpy as np
import pandas as pd
import faiss
import os
import cv2
import argparse
def load_train_dataset():
  x_train,y_train,x_test,y_test = fashion_dataset()
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
  return x_train,y_train
def preprocess_image(image):
  #image = cv2.imread("/content/abc.jpg")
  resize_img = cv2.resize(image,(28,28))
  gray_img = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
  bitimg = cv2.bitwise_not(gray_img)
  reshape = bitimg.reshape(1,28,28,1)
  return reshape
def similar_images(MODEL_TYPE,query_image):
  if MODEL_TYPE=="resnet":
    model = getResNet50Model()
    vector_index=faiss.read_index("Database/faiss_resnet_index.index")
    test_norm_feature = extract_resnet_feature(model,query_image)
    distance,indices = search_database(vector_index,test_norm_feature,top_k = 5)
  elif MODEL_TYPE=="autoencoder":
    autoencoder = AutoencoderModel()
    encoder_model = autoencoder.load_model(AUTOENCODER_MODEL_PATH)
    vector_index=faiss.read_index("Database/faiss_autoencoder_index.index")
    test_norm_feature = extract_autoencoder_feature(encoder_model,query_image)
    distance,indices = search_database(vector_index,test_norm_feature,top_k = 5)
  elif MODEL_TYPE=="siamesenet":
    input_shape = (28,28,1)
    siamese_triplet_model = SiameseTripletModel(input_shape)
    siamese_model = siamese_triplet_model.load_model(SIAMESE_MODEL_PATH)
    vector_index=faiss.read_index("Database/faiss_siamese_index.index")
    test_norm_feature = extract_siamese_feature(siamese_model,query_image)
    distance,indices = search_database(vector_index,test_norm_feature,top_k = 5)
  return distance,indices

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model configuration and testing parameters.')
    parser.add_argument('-mt','--model_type', type=str, choices=['resnet', 'autoencoder', 'siamesenet', 'VIT'],
                          default='autoencoder', help='Type of model to use.')
    parser.add_argument('-i','--test_image_path', type=str, default='input/',
                          help='Path of the test the image file')
    parser.add_argument('-o','--output_path', type=str, default='output/',
                          help='saving the output of model')
    return parser.parse_args()
if __name__ == '__main__':
  args = parse_arguments()
  test_image_path = args.test_image_path
  output_path = args.output_path
  MODEL_TYPE = args.model_type
  x_train,y_train = load_train_dataset()
  for imgfile in os.listdir(test_image_path):
    image = cv2.imread(f"{test_image_path}/{imgfile}")
    query_image=preprocess_image(image)
    distance,indices = similar_images(MODEL_TYPE,query_image)
    plot_results_save(imgfile,output_path,query_image,distance,x_train,y_train,indices)
