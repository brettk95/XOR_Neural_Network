This simple neural network predicts the outcome of an XOR relationship given two variables for each input.
It has been created using only numpy and was done primarily to show how the backpropagation code works.
The backpropagation algorithm is capable of taking a whole matrix of inputs as opposed to a vector for every single input.
This makes the code a lot easier to read (and quicker too (???)) as it minimizes the use of for loops and the need to index.

# How the feedforward and backpropagation algorithm works in this implementation:

Backpropagation:
1.  calculate the rror in the last layer (dL)
2.  backpropagate the error until layer 1 (input layer = layer 0 - layers indexed as such to make it easier when coding, but normally 
