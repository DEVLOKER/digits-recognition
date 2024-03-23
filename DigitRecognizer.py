import os, io
# Set TF_ENABLE_ONEDNN_OPTS environment variable to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import json
from contextlib import redirect_stdout
import pathlib


root_dir = pathlib.Path(__file__).parent

class DigitRecognizer(object):

    input_shape = (28, 28, 1) # image dimension (width, height), channels (1)
    num_classes = 10 # output dimension (from 0 to 9)
    batch_size = 128 # output shape
    epochs = 10 # number of tests

    model_name = 'trained-model.keras'
    log_dir = pathlib.Path(root_dir, ".logs") # + datetime.now().strftime("%Y%m%d-%H%M%S")
    

    def __init__(self, modal_path=None):
        if(modal_path is None):
            self.build()
            self.train()
        else:
            self.load(modal_path)
        self.log()
        
    def build(self):
        self.model = DigitRecognizer.CNN() # Convolutional Neural Network
        # self.model = DigitRecognizer.MLP() # Multilayer Perceptron
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

    # Convolutional Neural Network (CNN)
    @staticmethod
    def CNN():
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Convolution2D(
            input_shape=DigitRecognizer.input_shape,
            kernel_size=5,
            filters=8,
            strides=1,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling()
        ))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))
        model.add(tf.keras.layers.Convolution2D(
            kernel_size=5,
            filters=16,
            strides=1,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling()
        ))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=DigitRecognizer.batch_size,
            activation=tf.keras.activations.relu
        ))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(
            units=DigitRecognizer.num_classes,
            activation=tf.keras.activations.softmax,
            kernel_initializer=tf.keras.initializers.VarianceScaling()
        ))
        return model

    # Multilayer Perceptron
    @staticmethod
    def MLP():
        model = tf.keras.models.Sequential(layers=[
            tf.keras.layers.Flatten(
                input_shape=DigitRecognizer.input_shape
            ),  # Use Input layer with shape argument
            tf.keras.layers.Dense(
                units=DigitRecognizer.batch_size,
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.l2(0.002) 
            ),  # Hidden layer with 128 neurons
            tf.keras.layers.Dense(
                units=DigitRecognizer.batch_size,
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.l2(0.002)
            ),  # Dropout layer for regularization
            tf.keras.layers.Dense(
                units=DigitRecognizer.num_classes,
                activation=tf.keras.activations.softmax
            ) # Output layer with 10 units for 10 classes
        ])
        return model


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
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=DigitRecognizer.log_dir, histogram_freq=1)

        timer_start = datetime.now()
        training = self.model.fit(
            x_train_normalized,
            y_train,
            epochs=self.epochs,
            validation_data=(x_test_normalized, y_test),
            callbacks=[tensorboard_callback] # , batch_size=DigitRecognizer.batch_size,verbose=1
        )
        timer_end = datetime.now()
        difference = timer_end - timer_start
        print("The model has successfully trained in {:2f} seconds.".format(difference.total_seconds()))

        train_loss, train_accuracy = self.model.evaluate(x_train_normalized, y_train)
        print('Training loss: {}, Training accuracy: {}'.format(train_loss, train_accuracy))
        validation_loss, validation_accuracy = self.model.evaluate(x_test_normalized, y_test)
        print('Validation loss: {}, Validation accuracy: {}'.format(validation_loss, validation_accuracy))

        self.model.save(DigitRecognizer.model_name)#, save_format='h5') # keras | h5
        print("Saving the model as {}".format(DigitRecognizer.model_name))

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
        plt.savefig(pathlib.Path(DigitRecognizer.log_dir, "training_history.jpg"))
        plt.show()

    def load(self, filename=model_name): # DigitRecognizer.model_name
        self.model = tf.keras.models.load_model(filename)
        return self.model
    
    def predict_digit(self, img):
        # prepare the image
        img_array = self.process_image(img)
        #predicting
        prediction = self.model.predict([img_array])[0]
        # print(pd.DataFrame(prediction))
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

    def log(self):
        tf.keras.utils.plot_model( self.model, to_file=pathlib.Path(DigitRecognizer.log_dir, "model_plot.jpg"), show_shapes=True, show_layer_names=True)
        self.model.summary()
        # model_config = self.model.get_config()
        model_json = json.loads(self.model.to_json())
        string_buffer = io.StringIO()
        with redirect_stdout(string_buffer):
            self.model.summary()
        model_summary = string_buffer.getvalue()#.replace('\t', '    ')

        f = open(pathlib.Path(DigitRecognizer.log_dir, "model_json.json"), "w", encoding='utf-8')
        f.write(json.dumps(model_json, ensure_ascii=False, indent=4))
        f.close()
        f = open(pathlib.Path(DigitRecognizer.log_dir, "model_summary.txt"), "w", encoding='utf-8')
        f.write(model_summary)
        f.close()

        # self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        # model_summary = open('model_summary.txt', 'r', encoding='utf-8').read()
        return model_json, model_summary

if __name__ == "__main__":
    # digit_recognizer = DigitRecognizer()
    digit_recognizer = DigitRecognizer(DigitRecognizer.model_name)
    # Load the image
    image_path = "digits/9.jpg"
    img = Image.open(image_path).convert('L')
    digit, accuracy, prediction = digit_recognizer.predict_digit(img)
    print(' digit : {} \n accuracy: {:.2f}%'.format(digit, accuracy*100))
    