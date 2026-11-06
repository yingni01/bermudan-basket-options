import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation

class NeuralNetwork():

  def __init__(self, nn_parameters):
    self.input_dim = nn_parameters["input_dim"]
    self.num_neurons = nn_parameters["num_neurons"]
    self.activation_function = nn_parameters["activation_function"]
    self.batch_size = nn_parameters["batch_size"]
    self.epochs1 = nn_parameters["epochs1"]
    self.epochs2 = nn_parameters["epochs2"]
    self.epochs = self.epochs1
    self.model = self.build_model()

  def build_model(self):
    # define the keras model

    model = Sequential()

    #model.add(Dense(self.input_dim+self.num_neurons, input_dim=self.input_dim, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    
    # First, add an Input layer
    model.add(Input(shape=(self.input_dim,)))

    # Then add your Dense (hidden) layer without input_dim
    model.add(Dense(self.input_dim + self.num_neurons, 
                kernel_initializer=tf.keras.initializers.GlorotNormal()))
    
    model.add(BatchNormalization())
    model.add(Activation(self.activation_function))
    
    # Second hidden layer 
    model.add(Dense(self.num_neurons, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    model.add(BatchNormalization())
    model.add(Activation(self.activation_function))
    
    # Third hidden layer
    model.add(Dense(self.num_neurons, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    model.add(BatchNormalization())
    model.add(Activation(self.activation_function))
    
    model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotNormal()))
    model.add(BatchNormalization())
    model.add(Activation('linear'))

    opt = tf.keras.optimizers.Adam()
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])

    return model

  def train(self, X, y, i, plot=False):
    history = self.model.fit(X, y, epochs=self.epochs, validation_split=0.02, batch_size=self.batch_size, verbose=False)
    self.epochs = self.epochs2
    if plot:
        # summarize history for loss
        plt.plot(history.history['loss'][20:])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig("training_loss" + str(i) + ".png", format="png")
        plt.close()

  def predict(self, X):
    return self.model(X).numpy()
