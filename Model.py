import os
# Set TF_ENABLE_ONEDNN_OPTS environment variable to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import datetime
from tools import *


class Model(object):

    input_shape = (28, 28, 1) # image dimension (width, height), channels (1)
    num_classes = 10 # output dimension (from 0 to 9)
    batch_size = 128
    epochs = 5 # number of tests

    model_name = 'trained-model.h5'

    def __init__(self, modal_path=None):
        if(modal_path is None):
            self.build()
            self.train()
        else:
            self.load(modal_path)
        # self.model.summary()
        # tf.keras.utils.plot_model( self.model, show_shapes=True, show_layer_names=True)

    def build(self):
        # self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Convolution2D(
        #     input_shape=Model.input_shape,
        #     kernel_size=5,
        #     filters=8,
        #     strides=1,
        #     activation=tf.keras.activations.relu,
        #     kernel_initializer=tf.keras.initializers.VarianceScaling()
        # ))
        # self.model.add(tf.keras.layers.MaxPooling2D(
        #     pool_size=(2, 2),
        #     strides=(2, 2)
        # ))
        # self.model.add(tf.keras.layers.Convolution2D(
        #     kernel_size=5,
        #     filters=16,
        #     strides=1,
        #     activation=tf.keras.activations.relu,
        #     kernel_initializer=tf.keras.initializers.VarianceScaling()
        # ))
        # self.model.add(tf.keras.layers.MaxPooling2D(
        #     pool_size=(2, 2),
        #     strides=(2, 2)
        # ))
        # self.model.add(tf.keras.layers.Flatten())
        # self.model.add(tf.keras.layers.Dense(
        #     units=128,
        #     activation=tf.keras.activations.relu
        # ))
        # self.model.add(tf.keras.layers.Dropout(0.2))
        # self.model.add(tf.keras.layers.Dense(
        #     units=10,
        #     activation=tf.keras.activations.softmax,
        #     kernel_initializer=tf.keras.initializers.VarianceScaling()
        # ))
        # self.model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #     loss=tf.keras.losses.sparse_categorical_crossentropy,
        #     metrics=['accuracy']
        # )

        self.model = tf.keras.Sequential([
            tf.layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=self.input_shape),
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes, activation="softmax")
        ])
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
        
        # self.model = model = keras.Sequential([
        #     layers.Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=self.input_shape),
        #     layers.MaxPooling2D(pool_size=(2, 2)),
        #     layers.Conv2D(64, (3, 3), activation='relu'),
        #     layers.MaxPooling2D(pool_size=(2, 2)),
        #     layers.Flatten(),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dropout(0.3),
        #     layers.Dense(64, activation='relu'),
        #     layers.Dropout(0.5),
        #     layers.Dense(self.num_classes, activation='softmax')
        # ])
        # self.model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
        
        # self.model = models.Sequential(layers=[
        #     layers.Input(shape=(28, 28)),  # Use Input layer with shape argument
        #     layers.Flatten(),  # Flatten the input : input_shape=(28, 28)
        #     layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
        #     layers.Dropout(0.2),  # Dropout layer for regularization
        #     layers.Dense(10, activation='softmax')  # Output layer with 10 units for 10 classes
        # ])
        # self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

    def train(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # reshape to have the missing color_channels dimensions
        x_train_with_chanels = x_train.reshape(x_train.shape + (1,))
        x_test_with_chanels = x_test.reshape(x_test.shape + (1, ))
        # normalized data
        x_train_normalized = x_train_with_chanels/255.
        x_test_normalized = x_test_with_chanels/255.
        # change data type to float 32 bits
        # x_train = x_train_normalized.astype(np.float32) 
        # x_test /= x_test_normalized.astype(np.float32)
        # Training the Model on Tensorflow
        log_dir=".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        training = self.model.fit(
            x_train_normalized,
            y_train,
            epochs=self.epochs,
            validation_data=(x_test_normalized, y_test),
            callbacks=[tensorboard_callback] # , batch_size=self.batch_size,verbose=1
        )
        print("The model has successfully trained")

        train_loss, train_accuracy = self.model.evaluate(x_train_normalized, y_train)
        print('Training loss: {}, Training accuracy: {}'.format(train_loss, train_accuracy))
        validation_loss, validation_accuracy = self.model.evaluate(x_test_normalized, y_test)
        print('Validation loss: {}, Validation accuracy: {}'.format(validation_loss, validation_accuracy))

        self.model.save(Model.model_name)#, save_format='h5') # keras | h5
        print("Saving the model as {}".format(Model.model_name))

        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Plot training & validation accuracy values
        ax1.plot(training.history['accuracy'])
        ax1.plot(training.history['val_accuracy'])
        ax1.set_title('Model accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper left')
        # Plot training & validation loss values
        ax2.plot(training.history['loss'])
        ax2.plot(training.history['val_loss'])
        ax2.set_title('Model loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def load(self, filename=model_name): # Model.model_name
        self.model = tf.keras.models.load_model(filename)
        # self.model.load_weights(filename)
        # Predictions in form of one-hot vectors (arrays of probabilities).
        # predictions_one_hot = loaded_model.predict([x_test_normalized])
        # pd.DataFrame(predictions_one_hot)
        return self.model
    
    def predict_digit(self, img):
        # prepare the image
        img_array = self.process_image(img)
        #predicting
        prediction = self.model.predict([img_array])[0]
        predicted_digit = np.argmax(prediction)
        accuracy = max(prediction)
        return predicted_digit, accuracy, prediction
    
    def process_image(self, img):
        #resize image to 28x28 pixels
        img = img.resize((28,28))
        #convert rgb to grayscale
        img = img.convert('L')
        img_array = np.array(img)
        img_array = img_array.reshape(1,28,28,1)
        img_array = img_array/255.0
        img_array = 1 - img_array
        return img_array


        # if not image.getbbox(): # Is the image empty
        #     return None
        
        # img = img.resize((28, 28))  # Resize to match model input shape
        # img_array = np.array(img) / 255.0  # Normalize pixel values
        # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # return img_array

        # img = replace_transparent_background(img)
        # img = trim_borders(img)
        # img = pad_image(img)
        # img = to_grayscale(img)
        # img = invert_colors(img)
        # img = resize_image(img, 28, 28)
        # img = scale_down_intensity(img)
        # img_array = np.array([ np.array(img).flatten() ])
        # return img_array
    

if __name__ == "__main__":
    # model = Model()
    model = Model(Model.model_name)
    # Load the image
    image_path = "digits/9.jpg"
    img = Image.open(image_path).convert('L')
    digit, accuracy, prediction = model.predict_digit(img)
    print(' digit : {} \n accuracy: {:.2f}%'.format(digit, accuracy*100))
    