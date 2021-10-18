import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import fashion_mnist

from generalizer import Generalizer


if __name__ == "__main__":

    # load and prepare fashion mnist dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


    # set hyperparameters
    optimizer = tf.keras.optimizers.Adam()
    beta = 2.0

    # set up metrics
    train_classifier_loss = tf.keras.metrics.Mean(name='train_classifier_loss')
    train_autoencoder_loss = tf.keras.metrics.Mean(name='train_autoencoder_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    # build model
    model = Generalizer()


    # define training
    @tf.function
    def train_step(images, labels):

        # classifier training step: train the classifier to output good classifications, 
        # while the autoencoder loss is high
        with tf.GradientTape() as tape:
            predictions, decoded = model(images, training=True)

            # loss: classifier loss - weighted autoencoder loss (so AE-loss reduces the loss if it is high)
            loss = model.classifier_loss(labels, predictions) - beta * model.autoencoder_loss(images, decoded)

            # get the relevant variables
            var = [v for v in model.trainable_variables if 'shared' in v.name or 'classifier' in v.name]

            # apply gradients only to those variables
            gradients = tape.gradient(loss, var)
            optimizer.apply_gradients(zip(gradients, var))

            # for plotting
            train_classifier_loss(model.classifier_loss(labels, predictions))
            train_accuracy(labels, predictions)

        # autoencoder training step: train the autoencoder to recreate the image
        with tf.GradientTape() as tape:
            predictions, decoded = model(images, training=True)

            # loss: autoecoder loss
            loss = model.autoencoder_loss(images, decoded)

            var = [v for v in model.trainable_variables if 'autoencoder' in v.name]

            # apply gradients only to those variables
            gradients = tape.gradient(loss, var)
            optimizer.apply_gradients(zip(gradients, var))

            train_autoencoder_loss(loss)
            train_accuracy(labels, predictions)


    
    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, decoded = model(images, training=False)
        t_loss = model.classifier_loss(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)




    EPOCHS = 50

    for epoch in range(EPOCHS):
        
        # Reset the metrics at the start of the next epoch
        train_classifier_loss.reset_states()
        train_autoencoder_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Classifier Loss: {train_classifier_loss.result()}, '
            f'Autoencoder Loss: {train_autoencoder_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )