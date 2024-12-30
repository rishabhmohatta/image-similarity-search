import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50  import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class getResNet50Model:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.resnet_model = ResNet50(weights='imagenet', input_shape=self.input_shape, include_top = True)
        self.output = self.resnet_model.get_layer('avg_pool').output
        # self.reduce_output = Dense(512, activation='relu', name='feature_extraction')(self.output)
        self.resnet_model = Model(self.resnet_model.input, self.output)
        # print(self.resnet_model.summary())
        

    '''
    Use Resnet50 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, images):
        # img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        # img = image.img_to_array(img)
        # img = np.expand_dims(img, axis=0)
        resized_images = tf.image.resize(images, (self.input_shape[0], self.input_shape[1]))
        image = np.repeat(resized_images,3,axis=-1)
        # print(image.shape)
        img = preprocess_input(image)
        feat = self.resnet_model.predict(img)
        # norm_feat = feat/np.linalg.norm(feat,axis=1, keepdims=True)
        return feat