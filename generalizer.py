import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras import Model

class Generalizer(tf.keras.Model):

    def __init__(self):
        super(Generalizer, self).__init__()

        self.classifier_loss = SparseCategoricalCrossentropy(from_logits=True)
        self.autoencoder_loss = MeanSquaredError()

        # shared layers
        self.conv1 = Conv2D(32, 3, activation='relu', kernel_regularizer='l2', name="shared_input")
        self.flatten = Flatten(name="shared_flatten")

        # (shared) high dimensional generalized layer
        self.d1 = Dense(128, activation='relu', name="shared_embedding")

        # classifier layers
        self.d2 = Dense(10, name="classifier_out")

        # autoencoder layers
        self.autoenc = Dense(784, activation='sigmoid', name="autoencoder_main")
        self.auto_out = Reshape((28, 28), name="autoencoder_out")


    def call(self, x):

        # shared forward pass
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)

        # classifier forward
        classifier_out = self.d2(x)
        
        # autoencoder forward
        autoenc = self.autoenc(x)
        autoencoder_out = self.auto_out(autoenc)


        return classifier_out, autoencoder_out
