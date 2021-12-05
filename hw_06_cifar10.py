"""
hw_06-cifar10.ipynb
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

"""

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch,
each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class.
The training batches contain the remaining images in random order,
but some training batches may contain more images from one class than another.
Between them, the training batches contain exactly 5000 images from each class.

Source: https://www.cs.toronto.edu/%7Ekriz/cifar.html
"""

train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)

for elem in train_ds:
  print(elem)
  break

print(len(train_ds))
print(len(test_ds))

def prepare_cifar_data(cifar_data):
  #convert data from uint8 to float32
  cifar_data = cifar_data.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
  cifar_data = cifar_data.map(lambda img, target: ((img/128.)-1., target))
  #create one-hot targets
  cifar_data = cifar_data.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  #cache this progress in memory, as there is no need to redo it; it is deterministic after all
  cifar_data = cifar_data.cache()
  #shuffle, batch, prefetch
  cifar_data = cifar_data.shuffle(len(cifar_data))
  cifar_data = cifar_data.batch(128)
  cifar_data = cifar_data.prefetch(20)
  #return preprocessed dataset
  return cifar_data

train_dataset = train_ds.apply(prepare_cifar_data)
test_dataset = test_ds.apply(prepare_cifar_data)

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[1, 1],
            padding='same',
            activation=tf.nn.relu
        )

        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )

        self.batch = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(tf.nn.relu)




    def call(self, x):
        x_out_1 = self.conv1(x)
        x_out_1 = self.batch(x_out_1, training=True)
        x_out_1 = self.activation(x_out_1)

        x_out_2 = self.conv3(x_out_1)
        x_out_2 = self.batch(x_out_2, training=True)
        x_out_2 = self.activation(x_out_2)

        x_out_3 = self.conv3(x_out_2)
        x_out_3 = self.batch(x_out_3, training=True)
        x_out_3 = self.activation(x_out_3)

        x_out_3 = tf.keras.layers.Concatenate(axis=-1)([x, x_out_3])

        return x_out_3

class ResNet(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )

        self.resBlock1 = ResidualBlock()

        self.resBlock2 = ResidualBlock()

        self.resBlock3 = ResidualBlock()

        self.resBlock4 = ResidualBlock()

        self.globalPool = tf.keras.layers.GlobalAveragePooling2D()


        self.dense1 = tf.keras.layers.Dense(10)



    def call(self, x):
        x = self.conv1(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = self.globalPool(x)
        x = self.dense1(x)
        out = tf.nn.softmax(x)

        return out

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool3 = tf.keras.layers.GlobalAveragePooling2D()
        #self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 32,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = tf.nn.softmax(x)
        return out

class TransitionalLayer(tf.keras.layers.Layer):
  def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[1, 1],
            padding='valid',
            activation=tf.nn.relu
        )

        self.activation = tf.keras.layers.Activation(tf.nn.relu)

        self.batch = tf.keras.layers.BatchNormalization()

        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(1, 1),
            padding='valid'
        )

  def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)
        x = self.batch(x)
        out = self.pool(x)

        return out

class DenseBlock(tf.keras.layers.Layer):
  def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[1, 1],
            padding='valid',
            activation=tf.nn.relu
        )

        self.activation = tf.keras.layers.Activation(tf.nn.relu)

        self.batch = tf.keras.layers.BatchNormalization()

        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(1, 1),
            padding='valid'
        )

  def call(self, inputs):
        inputs = self.conv1(inputs)
        x = self.batch(inputs)
        x = self.activation(x)

        x = self.conv1(x)
        x = self.batch(x)
        x = self.activation(x)

        x = self.conv1(x)

        out = tf.keras.layers.Concatenate(axis=-1)([inputs, x])

        return out

class DenseNet(tf.keras.Model):
  def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )

        self.denseBlock1 = DenseBlock()
        self.denseBlock2 = DenseBlock()
        self.denseBlock3 = DenseBlock()

        self.transition1 = TransitionalLayer()
        self.transition2 = TransitionalLayer()
        self.transition3 = TransitionalLayer()

        self.outLayer = tf.keras.layers.Dense(units=10)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()

  def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        x = self.denseBlock1(x)
        x = self.transition1(x)

        x = self.denseBlock2(x)
        x = self.transition2(x)

        x = self.denseBlock3(x)
        x = self.transition3(x)

        x = self.pool(x)
        out = self.outLayer(x)
        out = tf.nn.softmax(out)

        return out

def train_step(model, input, target, loss_function, optimizer):
  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
    prediction = model(input)
    loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test(model, test_data, loss_function):
  # test over complete test data

  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model(input)
    sample_test_loss = loss_function(target, prediction)
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy

tf.keras.backend.clear_session()

### Hyperparameters
num_epochs = 30
learning_rate = 0.001

# Initialize the model.
model = DenseNet()
# Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Initialize lists for later visualization.
train_losses = []

test_losses = []
test_accuracies = []

#testing once before we begin
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)

    #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    #testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

"""With the same hyperparameters, the ResNet achieved an accuracy of 73% while taking around 20 minutes and the CNN of the week before achieved an accuracy of 53% while taking around 3 minutes.

The DenseNet achieved an accuracy of 62% with 30 epochs and taking around 18 minutes.
"""

import matplotlib.pyplot as plt

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1,line2, line3),("training","test", "test accuracy"))
plt.show()
