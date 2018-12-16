'''
very basic neural network to predict XOR gate outcomes
created mainly to demonstrate backpropagation using matrix of all inputs
(instead of a vector for each input)
'''

import numpy as np

class Network(object):
    
    def __init__(self,size):
        self.size = size
        self.layers = len(size)
        self.weights = [np.random.rand(y,x) for x,y in zip(size[:-1],size[1:])]
        self.biases = [np.random.rand(x,1) for x in size[1:]]
        
    def compilenet(self,training_data, eta, epoch):
        x,y = list(zip(*training_data))
        x,y = np.array(x),np.array(y)
        
        for i in range(epoch):
            yhat = self.forward(x) # vector of outputs after feedforwarding
            dL = self.outputerror(yhat,y) # vector of output errors
            errors = [] # a list to hold all errors
            
            errors.append(self.backprop(dL))
            errors.append(dL)
            
            # gradient of the cost function for each layer is calculated in the same step as the change in weights and biases in each layer
            self.weights[1] = self.weights[1] - (eta/len(x)) * np.dot(dL,self.a[0].T)
            self.weights[0] = self.weights[0] - (eta/len(x)) * np.dot(errors[0],x)
            
            self.biases[1] = self.biases[1] - (eta/len(x)) * dL
            self.biases[0] = self.biases[0] - (eta/len(x)) * errors[0]

        
    def backprop(self,dL):
        # backpropagate the error from the last layer (dL)
        # there are 3 layers in total, and only layers 1 and 2 have errors
        # in this case layer 2 = dL, so only need to calculate dL1
        
        dl1 = np.dot(self.weights[1].T, dL) * self.sigmoidprime(self.z[0]) # matrix of errors for layer 1 for matrix of inputs
        
        return dl1
    
    def forward(self,x):
        self.z = [] # matrix of all z values
        self.a = [] # matrix of all activation values
        
        z1 = np.dot(self.weights[0], x.T) #+ self.biases[0]
        self.z.append(z1)
        a1 = self.sigmoid(z1)
        self.a.append(a1)
        
        z2 = np.dot(self.weights[1], a1) #+ self.biases[1]
        self.z.append(z2)
        a2 = self.sigmoid(z2)
        self.a.append(a2)
        
        return a2
    
    def outputerror(self,yhat,y):
        return yhat - y.T
            
    def costfunction(self,a,y):
        error = np.sum((a - y) ** 2)/len(a)
        return error
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def sigmoidprime(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
        
    def costderivative(self,a,y):
        return a-y

# set up the network
net = Network([2,3,1])

# training data
x_train = np.array([[1,1],[1,0],[0,1],[0,0]])
y_train = np.array([[0],[1],[1],[0]])
training_data = list(zip(x_train,y_train))

# compile network
net.compilenet(training_data, 0.3,5000)

# test data
x_test = np.array([[0,1],[1,0],[1,1],[0,0]])
result = net.forward(x_test)
print(result)
