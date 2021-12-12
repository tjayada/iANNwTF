"""hw_07-LSTM.ipynb"""

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import numpy as np

def integration_task(seq_len, num_samples):

    for _ in range(num_samples):

        x = np.random.normal(size=seq_len)

        if np.sum(x) >= 1:
            target = 1
        else:
            target = 0

        x = np.expand_dims(x, -1)
        target = np.expand_dims(target, -1)

        yield x, target

def my_integration_wrapper():
    len = 5
    num = 80000
    r = integration_task(len, num)
    for i in r:
      yield i

data = tf.data.Dataset.from_generator(my_integration_wrapper, output_signature=(
                                            tf.TensorSpec(shape=(5,1), dtype= tf.float64),
                                            tf.TensorSpec(shape=(1,), dtype=tf.int16)))

train_ds = data.take(72000)
test_ds = data.skip(72000).take(8000)

def prepare(ds):

    # Prepare data for model

    # cache
    ds = ds.cache()
    # shuffle, batch, prefetch our dataset
    ds = ds.shuffle(5000)
    ds = ds.batch(64)
    ds = ds.prefetch(128)
    return ds

# prepare data
train_ds = train_ds.apply(prepare)
test_ds = test_ds.apply(prepare)

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Activation, GlobalAvgPool2D

class LSTM_Cell(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(LSTM_Cell, self).__init__()

        self.units = units

        # forget gate
        self.f_gate = Dense(units=self.units, activation="sigmoid")

        # input gate
        self.i_gate = Dense(units=self.units, activation="sigmoid")

        # candidate gate
        self.c_gate = Dense(units=self.units, activation="tanh")

        # output gate
        self.o_gate = Dense(units=self.units, activation="sigmoid")

        #hidden gate
        self.h_gate = Dense(units=self.units, activation="tanh", bias_initializer='ones')

        def call(self, x, states):

            # current cell state and hidden cell state
            hidden_state, cell_state = states

            # put input and hidden state in a tensor
            input = tf.concat([x, hidden_state], axis=0)

            # compute forget gate output
            forget = self.f_gate(input)

            # update the currentcell state
            cell_state = tf.matmul(forget, cell_state)

            # compute candidate gate output
            candidate = self.c_gate(input)

            # update hidden state: ((input, hiddenstate) x candidate output) + cell_state
            hidden_state = tf.matmul(input, candidate) + cell_state

            output = self.o_gate(hidden_state)

            return [hidden_state, output]

class LSTM_Layer(LSTM_Cell):
    def __init__(self, cell):
        super(LSTM_Layer, self).__init__()
        self.cell = cell

        self.layers = []

        for _ in range(cell):
            self.layers.append(LSTM_Cell())

    def call(self, x, states):
        outputs = []

        c_and_h = self.layers[0].call(x, states)

        outputs.append(c_and_h[1])

        for i in self.layers[1:]:
          for length in range(5):
            # not sure if it should be c_and_h
            c_and_h = i.call(x[length],(c_and_h[0],c_and_h[1]))
            outputs.append(c_and_h[1])

        return outputs

    def zero_states(self, batch_size):

        return tf.zeros([batch_size, self.cell]), tf.zeros([batch_size, self.cell])

class LSTM_Model(tf.keras.Model):

    def __init__(self):
        super(LSTM_Model, self).__init__()

        self.read_in = tf.keras.layers.Dense(units=64, activation='tanh', bias_initializer='ones')

        self.lstm_layer = LSTM_Layer(cell=10)

        self.out = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, x):
        init = (self.lstm_layer.zero_states(batch_size=64))
        x = self.read_in(x)
        x = self.lstm_layer.call(x, init)
        x = tf.expand_dims(x[-1], -1)
        x = self.out(x)

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
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy

train_dataset = train_ds
test_dataset = test_ds

### Hyperparameters
num_epochs = 20
learning_rate = 0.001

# Initialize the model.
model = LSTM_Model()
# Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer: Nadam with default parameters
optimizer = tf.keras.optimizers.Nadam(learning_rate)

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

# We train for num_epochs.
for epoch in range(num_epochs):
    print(f'Epoch {str(epoch)}: accuracy of testdata at {test_accuracies[epoch]}')
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
print('Final accuracy:', test_accuracies[num_epochs])
