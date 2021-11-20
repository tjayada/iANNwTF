import pandas as pd
import tensorflow as tf
import numpy as np

data_set = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/winequality-red.csv", sep = ";")

data_set.head()

data_set.describe()

data_set = (data_set-data_set.min())/(data_set.max()-data_set.min())

data_set.head()
median = data_set[["quality"]].median()
print(median)

# there are different categories like "fixed acidity" or "density" which then have different
# values according to the wine tester. The values are in order of the categories, so the first
# value responds to the first categorie and so on.
# we try to take these categories as input and try to predict the last categorie, the quality.

train_ds = data_set.sample(n = 1000)
data_set = data_set.drop(train_ds.index)

test_ds = data_set.sample(n = 400)
data_set = data_set.drop(test_ds.index)

validation_ds = data_set.sample(frac=1)

print(len(train_ds))

print(len(test_ds))

print(len(validation_ds))

train_ds_labels = train_ds[['quality']]
train_ds = train_ds.drop("quality", axis=1)

test_ds_labels = test_ds[['quality']]
test_ds = test_ds.drop("quality", axis=1)

validation_ds_labels = validation_ds[['quality']]
validation_ds = validation_ds.drop("quality", axis=1)

train_ds_labels.head()

train_ds.head()

def make_binary(target):
  if target >= median:
    return 1
  else: return 0

train_tensor = tf.data.Dataset.from_tensor_slices(train_ds)
train_labels_tensor = tf.data.Dataset.from_tensor_slices(train_ds_labels)
train_labels_tensor = train_labels_tensor.map(make_binary)
train_tensor = tf.data.Dataset.zip((train_tensor, train_labels_tensor))

test_tensor = tf.data.Dataset.from_tensor_slices(test_ds)
test_labels_tensor = tf.data.Dataset.from_tensor_slices(test_ds_labels)
test_labels_tensor = test_labels_tensor.map(make_binary)
test_tensor = tf.data.Dataset.zip((test_tensor, test_labels_tensor))

validation_tensor = tf.data.Dataset.from_tensor_slices(validation_ds)
validation_labels_tensor = tf.data.Dataset.from_tensor_slices(validation_ds_labels)
validation_labels_tensor = validation_labels_tensor.map(make_binary)
validation_tensor = tf.data.Dataset.zip((validation_tensor, validation_labels_tensor))

def preprocessing_wine_data(data):
  # dont need to change data type from int64 to float32
  ds_wine = data
  # create binary encoding
  # done before zipping above calling the map function with the make_binary function
  # cache progress
  ds_wine = ds_wine.cache()
  # batch data
  ds_wine = ds_wine.shuffle(1000)
  ds_wine = ds_wine.batch(32)
  ds_wine = ds_wine.prefetch(20)

  return ds_wine

train_tensor = preprocessing_wine_data(train_tensor)

test_tensor = preprocessing_wine_data(test_tensor)

validation_tensor = preprocessing_wine_data(validation_tensor)

# Custom Model
class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = SimpleDense(190, activation = tf.nn.sigmoid)
        self.dense2 = SimpleDense(190, activation = tf.nn.sigmoid)
        self.out = SimpleDense(1, activation = tf.nn.sigmoid)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x, training=True)
        out = self.out(x)
        return out

class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units, activation):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        return x

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
    sample_test_accuracy =  np.round(target, 0) == np.round(prediction, 0)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy

#tf.keras.backend.clear_session()

#For showcasing we only use a subset of the training and test data (generally use all of the available data!)
train_dataset = train_tensor
test_dataset = test_tensor

### Hyperparameters
num_epochs = 11
learning_rate = 0.15

# Initialize the model.
model = MyModel()
# Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
# Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
optimizer = tf.keras.optimizers.SGD(learning_rate)

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

validate = validation_tensor
validate_accuracies = []
for elem in validate:
    validate_loss, validate_accuracy = test(model, validation_tensor, cross_entropy_loss)
    validate_accuracies.append(validate_accuracy)

plt.figure()
line1, = plt.plot(validate_accuracies)
plt.xlabel("Data Points")
plt.ylabel("Accuracy")
plt.show()
