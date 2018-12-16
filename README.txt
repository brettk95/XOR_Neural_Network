This simple neural network predicts the outcome of an XOR relationship given two variables for each input.
It has been created using only numpy and was done primarily to show how the backpropagation code works.
The backpropagation algorithm is capable of taking a whole matrix of inputs as opposed to a vector for every single input.
This makes the code a lot easier to read (and quicker too (???)) as it minimizes the use of for loops and the need to index.



# How the feedforward and backpropagation algorithm works in this implementation:

Input set X = (x1 ... xn)

Feedforward:
1. Feedforward the input set X
2. Keep an array of the z and activation values for each layer (these will be used in the backpropagation stages)

Backpropagation:
1.  calculate the error in the last layer (dL)
2.  backpropagate the error until layer 1 (*1) 
3.  calculate the gradients of the cost function for each layers
4.  change the weights and biases and each layer using the gradients calculated in previous step



# Problems & Possible Causes:
1.  The learning rate is extremely slow (it takes about 4000 epochs for the network to learn how to predict a 2D input)
The activation function used (sigmoid) may not be very good for this particular scenario. The outputs are either 0 or 1, and the gradient of the sigmoid function is very small - thus slow learning? - as the y value approaches 0 or 1. Perhaps using a ReLU activation function could increase learning rate.

2. Addition of bias confuses the network
Will need to further reasearch why having a bias confuses the network.



(*1):
input layer = layer 0
layers indexed as such to make it easier when coding using zero indexing
normally the input layer is counted as layer 1
