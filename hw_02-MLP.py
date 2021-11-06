import numpy as np 
import matplotlib.pyplot as plt

# Our data
# all possible input values for the logical gates
inputs = np.array([[0,0],
                   [1,0],
                   [0,1],
                   [1,1]])

# all the logical gates
and_labels = np.array([[0],
                       [0],
                       [0],
                       [1]])

or_labels = np.array([[0],
                      [1],
                      [1],
                      [1]])

not_and_labels = np.array([[1],
                           [1],
                           [1],
                           [0]])

not_or_labels = np.array([[1],
                          [0],
                          [0],
                          [0]])

xor_labels = np.array([[0],
                       [1],
                       [1],
                       [0]])


# the sigmoid activation function
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

# the derivative of the sigmoid function
def sigmoidprime(x):
    return sigmoid(x) * (1 - sigmoid(x))



# the perceptron class
class Perceptron:
    """
    Perceptron class

    This class simulates a single Perceptron, which
    will be the building block for our Multi-Layer-Perceptron

    It takes input_units for the number of weights as an integer as
    well ass the learning rate alpha as arguments.
    """
    def __init__(self, input_units, alpha = 1):
        self.weights = input_units
        self.alpha = alpha
        self.bias = float(np.random.rand(1))
        self.drive = None

        weights_list = []
        for i in range(self.weights):
            weights_list.append(float(np.random.rand(1)))
    
        self.weights = weights_list

    def forward_step(self, inputs):
        self.inputs = inputs
        self.activation = self.weights @ self.inputs + self.bias
        return(sigmoid(self.activation))

    def update(self, delta):
        gradient_weights = delta * self.inputs

        self.weights -=  self.alpha * gradient_weights
        self.bias -= self.alpha * delta



class MLP():
    """
    A Multi-Layer-Perceptron 
    
    It is made of multiple instances of the Perceptron class above
    """

    def __init__(self):
        self.output = None
        self.inputs = None
        self.labels = None
        self.hidden_output = np.zeros([4])
        
        self.hiddenPerceptron1 = Perceptron(2)
        self.hiddenPerceptron2 = Perceptron(2)
        self.hiddenPerceptron3 = Perceptron(2)
        self.hiddenPerceptron4 = Perceptron(2)
        self.hiddenLayer = [self.hiddenPerceptron1, self.hiddenPerceptron2, self.hiddenPerceptron3, self.hiddenPerceptron4]
        
        self.outputPerceptron = Perceptron(4)

    def forward_step(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
        for i in range(len(self.hiddenLayer)):
            self.hidden_output[i] = self.hiddenLayer[i].forward_step(self.inputs)
        
        output_layer = self.outputPerceptron.forward_step(self.hidden_output)
        output_layer = np.reshape(output_layer, newshape=(-1))
        self.output = output_layer

    def backprop_step(self):
        delta_hidden = np.zeros([int(len(self.hiddenLayer))])
        for i in range(2):
            if i == 0:
                delta = (-(self.labels - self.output) *  sigmoidprime(self.outputPerceptron.activation))
                self.outputPerceptron.update(delta)
            elif i == 1:
                for e in range(len(self.hiddenLayer)):
                    delta_hidden[e] = delta * self.outputPerceptron.weights[e] * sigmoidprime(self.hiddenLayer[e].activation)
                    self.hiddenLayer[e].update(delta_hidden[e])



multi_pulti = MLP()

list_of_epochs=[]
list_of_losses=[]
list_of_accuracies=[]

for epoch in range(1000):
    list_of_epochs.append(epoch)
    
    accuracy_buffer = 0
    loss_buffer = 0
    
    for i in range(4):
        x = inputs[i]
        t = and_labels[i]
        
        multi_pulti.forward_step(x,t)
        multi_pulti.backprop_step()
        
        accuracy_buffer += int(float(multi_pulti.output>=0.5) == t)
        loss_buffer += (t-multi_pulti.output)**2
        
    list_of_accuracies.append(accuracy_buffer/4.0)
    list_of_losses.append(loss_buffer)



# Visualize the training progress. Loss and accuracy.
plt.figure()
plt.plot(list_of_epochs,list_of_losses)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(list_of_epochs,list_of_accuracies)
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.show()

