import numpy as np
from flask import Flask
from flask_ask import Ask, statement, question
import requests
import time
import unidecode
import json
from PIL import Image

from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.layers.dropout import dropout
from mynn.activations.relu import relu
from mynn.initializers.glorot_uniform import glorot_uniform
from mygrad.nnet.layers import max_pool
from mynn.losses.cross_entropy import softmax_cross_entropy

from mygrad import Tensor

app = Flask(__name__)
ask = Ask(app, '/')

class Model:
    ''' A simple convolutional neural network. '''
    def __init__(self):
        params = np.load("params.npy")
        
        #this gain is a parameter for the weight initializiation function glorot_uniform
        #which you can read more about in the documentation, but it isn't crucial for now
        #If you would like to read more about how Xavier Glorot explains the rationalization behind these weight initializations,
        #look here for his paper written with Yoshua Bengio. (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
        init_kwargs = {'gain': np.sqrt(2)}
        
        #We will use a dropout probability of 0.5 so that values are randomly set to 0 in our data
        self.dropout_prob = 0.5
        
        #initialize your two dense and convolution layers as class attributes using the functions imported from MyNN
        #We will use weight_initializer=glorot_uniform for all 4 layers
        
        #You know the input size of your first convolution layer. Try messing around with the output size but make sure that the following
        #layers dimmensions line up. For your first convolution layer start with input = 1, output = 20, filter_dims = 5, stride = 5,
        #padding = 0
        
        
        self.dense1 = dense(180, 200, weight_initializer = glorot_uniform, weight_kwargs = init_kwargs)
    
        self.dense2 = dense(200, 10, weight_initializer = glorot_uniform, weight_kwargs = init_kwargs)
        
        self.conv1 = conv(1, 20, (5, 5), stride = 1, padding = 0, weight_initializer = glorot_uniform, weight_kwargs = init_kwargs)
    
        self.conv2 = conv(20, 20, (2, 2), stride = 2, padding = 0, weight_initializer = glorot_uniform, weight_kwargs = init_kwargs)
        
        self.dropout = dropout(self.dropout_prob)
        
        self.conv1.weight = Tensor(params[0])
        self.conv1.bias = Tensor(params[1])
        
        self.conv2.weight = Tensor(params[2])
        self.conv2.bias = Tensor(params[3])
        
        self.dense1.weight = Tensor(params[4])
        self.dense1.bias = Tensor(params[5])
        
        self.dense2.weight = Tensor(params[6])
        self.dense2.bias = Tensor(params[7])
    
    
    def __call__(self, x):
        ''' Defines a forward pass of the model.
        
        Parameters
        ----------
        x : numpy.ndarray, shape=(N, 1, 28, 28)
            The input data, where N is the number of images.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, 10)
            The class scores for each of the N images.
        
        Pseudo-code
        -----------
        >>> create dropout object
        >>> compute the first convolutional layer by doing x.conv1
        >>> Perform ReLU by using relu(x)
        >>> Perform dropout by using x.dropout()
        >>> Use max_pool(x, size_pool, stride) to perform the pooling layer
        >>> repeat once 
        >>> perform two dense layers with ReLU dropout in between
        '''
        
        #first conv layer
        x = self.conv1(x)
        x = relu(x)
        x = self.dropout(x)
        x = max_pool(x, (2, 2), 2)
        
        #second conv layer
        x = self.conv2(x)
        x = relu(x)
        x = self.dropout(x)
        x = max_pool(x, (2, 2), 2)
        
        #performing the two dense layers
        x = x.reshape(x.shape[0], -1)
        
        x = self.dense1(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        
        return x

    @property
    def parameters(self):
        ''' A convenience function for getting all the parameters of our model. '''
        #create a list of every parameter contained in the 4 layers you wrote in your __init__ function
        #these layers contain all of your 
        return self.conv1.parameters + self.conv2.parameters + self.dense1.parameters + self.dense2.parameters

@app.route('/')
def homepage():
    return "Hello"

@ask.launch
def start_skill():
    welcome_message = 'Hello there, do you want to draw a number for me to classify?'
    return question(welcome_message)


def image_predict(image_name):
    model = Model()
    I = np.asarray(Image.open(image_name))[...,:-1]
    I = (np.sum(I, axis=2) / 3)[np.newaxis,np.newaxis]/255
    prediction = model(I)
    return np.argmax(prediction.data, axis=1)


@ask.intent("YesIntent")
def draw_number():
    wait_message = "Please start drawing a one digit number. Say done when you are done."
    return question(wait_message)

@ask.intent("DoneIntent")
def give_number():
    number = image_predict("canvas.png")
    number_msg = 'Did you draw the number {} ?'.format(number)
    return statement(number_msg)

@ask.intent("NoIntent")
def no_intent():
    bye_text = 'Okay, goodbye'
    return statement(bye_text)

if __name__ == '__main__':
    app.run(debug=True)