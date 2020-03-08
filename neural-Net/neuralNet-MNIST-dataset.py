from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)

network = models.Sequential()
# Dense(512) is a fully-connected layer with 512 hidden units.
# in the first layer, you must specify the expected input data shape :
# here, 28 X 28=784 -dimensional vectors.
network.add(layers.Dense(612, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(350, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255.

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255.

train_labels = to_categorical(train_labels)
print(train_labels)
test_labels = to_categorical(test_labels)
print(test_labels)

history = network.fit(train_images, train_labels, epochs=5, batch_size=24)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test Accuracy: {:.5f} '.format(test_acc))
print(test_acc)

# summarize history for accuracy
epochs = np.arange(5)
plt.plot(epochs, history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(epochs, history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
