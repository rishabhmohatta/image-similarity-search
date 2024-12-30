import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AutoencoderModel:
    def __init__(self):
        self.autoencoder = None
        self.encoder = None

    def build_autoencoder(self):
        """Define the architecture of the autoencoder."""
        input_img = layers.Input(shape=(28, 28, 1))
        
        # Encoder
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        # Decoder
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = models.Model(input_img, decoded)
        self.encoder = models.Model(input_img, encoded)
        print(self.autoencoder.summary())
        return self.autoencoder, self.encoder

    # Training model
    def train_autoencoder(self, x_train, x_val, Model_path, batch_size, epoch):
        print("training started")
        autoencoder, encoder = self.build_autoencoder()
        datagen = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     fill_mode='nearest')
        
        train_images = x_train[:2000]
        test_data = x_val[:400]
        print(test_data.shape)
        datagen.fit(train_images)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.fit(datagen.flow(train_images, train_images, batch_size=batch_size),
                        validation_data=(test_data, test_data),
                        epochs=epoch)
        autoencoder.save(Model_path)

    def load_model(self,model_path):
        encoder = models.load_model(model_path)
        # as 4 layer of autoencoder is encoder model output
        encoder_model = models.Model(inputs=encoder.input, outputs=encoder.layers[4].output)
        return encoder_model
    
    # def extract_feature(self,model,images):
    #     feature = model.predict(images)
    #     return feature