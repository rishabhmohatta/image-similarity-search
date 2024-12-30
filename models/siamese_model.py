import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class SiameseTripletModel:
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.triplets = None
        self.model = self.create_siamese_model()

    def create_triplets(self, images, labels):
        triplets = []
        label_to_images = {}
        
        for img, label in zip(images, labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img)
        
        for label, imgs in label_to_images.items():
            for i in range(len(imgs)):
                anchor = imgs[i]
                positive = imgs[np.random.choice(np.delete(np.arange(len(imgs)), i))]
                negative_label = np.random.choice(list(label_to_images.keys()))
                negative = label_to_images[negative_label][np.random.randint(len(label_to_images[negative_label]))]
                triplets.append((anchor, positive, negative))
        
        return np.array(triplets)

    def create_siamese_model(self):
        input_layer = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        model = models.Model(inputs=input_layer, outputs=x)
        return model

    def triplet_loss(self, y_true, y_pred):
        anchor, positive, negative = tf.split(y_pred, 3, axis=0)
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + 0.2, 0))

    def train(self, images, labels,model_path,batch_size=198, epochs=10):
        self.triplets = self.create_triplets(images, labels)
        num_triplets = len(self.triplets)
        if num_triplets % 3 != 0:
            self.triplets = self.triplets[:num_triplets - (num_triplets % 3)]
        
        anchors = np.array([t[0] for t in self.triplets])
        positives = np.array([t[1] for t in self.triplets])
        negatives = np.array([t[2] for t in self.triplets])
        
        siamese_model = models.Model(inputs=self.model.input, outputs=self.model.output)
        siamese_model.compile(optimizer='adam', loss=self.triplet_loss)
        
        # Train the model
        siamese_model.fit(
            np.vstack((anchors, positives, negatives)),
            np.zeros((len(self.triplets) * 3, 1)),  
            batch_size=batch_size,
            epochs=epochs
        )
        siamese_model.save(model_path)
    
    def load_model(self,model_path):
        siamese_model = models.load_model(model_path,custom_objects={'triplet_loss': self.triplet_loss})

        return siamese_model