# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# Importing the dataset
from tensorflow.keras.datasets import fashion_mnist

# Loading the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# Reshaping the dataset
x_train = tf.reshape(x_train, (-1, 28 * 28))
x_test = tf.reshape(x_test, (-1, 28 * 28))
# Shape of the dataset
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# Normalizing the dataset
x_train = x_train / 255.0
x_test = x_test / 255.0
# Building the model
model = keras.Sequential()
# Adding the input layer and the first hidden layer
model.add(keras.Input(shape=(784, )))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
# Compiling the model
model.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
# Training the model
history = model.fit(
    x_train,
    y_train,
    validation_data=(
        x_test,
        y_test,
    ),
    batch_size=256,
    epochs=100,
    verbose=2,
)
# Plotting the loss and accuracy
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend()
plt.plot(history.history["sparse_categorical_accuracy"],
         label="train_accuracy")
plt.plot(history.history["val_sparse_categorical_accuracy"],
         label="val_accuracy")
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.legend()
