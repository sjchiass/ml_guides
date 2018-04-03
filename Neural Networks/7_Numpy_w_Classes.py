import numpy as np

class IdentityActivation:
    def __init__(self, predecessor):
        # Remember what precedes this layer
        self.predecessor = predecessor
        # The activation function keeps the dimensions of its predecessor
        self.input_size = self.predecessor.output_size        
        self.output_size = self.input_size
        # Create an empty matrix to store this layer's last activation
        self.activation = np.zeros(self.output_size)
        # Initialize weights, if necessary
        self.init_weights()
        # Will never require a gradient
        self.require_gradient = False
    # This activation function has no parameters, so pass
    def init_weights(self):
        pass
    # It also has to gradients, so pass here too
    def zero_grad(self):
        pass
    # During a forward pass, it just passes its input forward unmodified
    def forward(self, x, evaluate=False):
        # Save a copy of the activation
        self.activation = x
        return self.activation
    # During backrop it passes the delta on unmodified
    def backprop(self, delta, y):
        return delta
    # It has no parameters
    def report_params(self):
        return []


class InputLayer(IdentityActivation):
    def __init__(self, input_size):
        # The size here is determined by the data that's going to be used
        self.input_size = (0, input_size)
        self.output_size = self.input_size
        # Create an empty matrix to store this layer's last activation
        self.activation = np.zeros(self.output_size)


class SigmoidActivation(IdentityActivation):
    # During a forward pass, it just applies the sigmoid function
    def forward(self, x, evaluate=False):
        self.activation = 1.0/(1.0+np.exp(-x))
        return self.activation
    # During backprop, it passes the delta through its derivative
    def backprop(self, delta, y):
        return delta * self.activation * (1 - self.activation)

class Variable:
    def __init__(self, data):
        self.data = data
        self.shape = self.data.shape
        self.grad = np.zeros(self.shape)
    # Cheap way of zeroing its gradient
    def zero_grad(self):
        self.grad *= 0

class DenseLayer(IdentityActivation):
    def __init__(self, predecessor, hidden, use_bias=True, require_gradient=True, positive_params=False):
        # Remember what precedes this layer
        self.predecessor = predecessor
        self.input_size = self.predecessor.output_size
        self.hidden = hidden
        self.output_size = (0, self.hidden)
        # It is possible that we don't want a bias term
        self.use_bias = use_bias
        # If you need non-negative parameters
        self.positive_params = positive_params
        # Save its activation
        self.activation = np.zeros(self.output_size)
        # Initialize the weights and biases
        self.init_params()
        # Most of the time, this layer will use gradients to train itself
        # However, this can be disabled manually
        self.require_gradient = require_gradient
    def zero_grad(self):
        self.weight.zero_grad()
        if self.use_bias:
            self.bias.zero_grad()
    def init_params(self):
        size_measure = self.input_size[1]
        if self.positive_params:
            lower, upper = 0., 0.5
        else:
            lower, upper = -1., 1.
        # Weights are initialized by a normal distribution
        self.weight = Variable(
            np.sqrt(2/size_measure) * np.random.uniform(lower, upper, size=(self.input_size[1], self.hidden))
        )
        if self.use_bias:
            self.bias = Variable(
                np.sqrt(2/size_measure) * np.random.uniform(lower, upper, size=(1, self.hidden))
            )
    # The forward pass is a matrix multiplication, with optional bias
    def forward(self, x, evaluate=False):
        x = x @ self.weight.data
        if self.use_bias:
            x += self.bias.data
        self.activation = x
        return self.activation
    # The delta just needs to be multipled by the layer's weight
    def backprop(self, delta, y):
        # Only calculate gradients if it's required
        if self.require_gradient:
            # The weight update requires the previous layer's activation
            self.weight.grad += self.predecessor.activation.transpose() @ delta
            # The bias update requires the delta to be "squished"
            # This can be done by multiplying by a vector of 1s
            if self.use_bias:
                self.bias.grad += np.ones((1, delta.shape[0])) @ delta
        return delta @ self.weight.data.transpose()
    # This DenseLayer is the first example of a layer with parameters
    def report_params(self):
        if self.use_bias:
            return [self.bias, self.weight]
        else:
            return [self.weight]

class SigmoidNLL(DenseLayer):
    # The forward pass is a matrix multiplication, with optional bias
    def forward(self, x, evaluate=False):
        # The feed forward starts off normal
        x = x @ self.weight.data
        if self.use_bias:
            x += self.bias.data
        # It changes when we apply the sigmoid
        self.activation = 1.0/(1.0+np.exp(-x))
        return self.activation
    # The delta is started here
    def backprop(self, delta, y):
        # Starting the delta
        delta = self.activation - y
        # The update is the same as a DenseLayer
        self.weight.grad += self.predecessor.activation.transpose() @ delta
        if self.use_bias:
            self.bias.grad += np.ones((1, delta.shape[0])) @ delta
        # The delta is passed backwards like a Denselayer
        return delta @ self.weight.data.transpose()

class SoftmaxCrossEntropy(DenseLayer):
    # The forward pass is a matrix multiplication, with optional bias
    def forward(self, x, evaluate=False):
        # The feed forward starts off normal
        x = x @ self.weight.data
        if self.use_bias:
            x += self.bias.data
        # It changes when we apply the sigmoid
        softmax_sum = np.exp(x) @ np.ones((self.hidden, 1))
        self.activation = np.exp(x) / softmax_sum
        return self.activation
    # The delta is started here
    def backprop(self, delta, y):
        # Starting the delta
        delta = self.activation - y
        # The update is the same as a DenseLayer
        self.weight.grad += self.predecessor.activation.transpose() @ delta
        if self.use_bias:
            self.bias.grad += np.ones((1, delta.shape[0])) @ delta
        # The delta is passed backwards like a Denselayer
        return delta @ self.weight.data.transpose()

# This class stores a list of layers for training
class NeuralNetwork:
    def __init__(self):
        self.layers = []
    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    # When you want the model to know you're not training
    def evaluate(self, x):
        for layer in self.layers:
            x = layer.forward(x, evaluate=True)
        return x
    def back_propagation(self, y):
        delta = np.zeros((0,0))
        for layer in reversed(self.layers):
            delta = layer.backprop(delta, y)
    def step(self,lr):
        for layer in self.layers:
            layer.update(lr)
    def zero_gradients(self):
        for layer in self.layers:
            layer.zero_grad()

class SGDOptimizer:
    def __init__(self, list_of_layers):
        self.list_of_layers = list_of_layers
        self.list_of_variables = []
        for layer in self.list_of_layers:
            self.list_of_variables += layer.report_params()
    def step(self, lr):
        for variable in self.list_of_variables:
            variable.data -= lr * variable.grad

class AdaGradOptimizer:
    def __init__(self, list_of_layers):
        self.list_of_layers = list_of_layers
        self.list_of_variables = []
        for layer in self.list_of_layers:
            self.list_of_variables += layer.report_params()
        self.gradient_histories = dict()
        for variable in self.list_of_variables:
            self.gradient_histories[variable] = np.ones(variable.shape)
    def step(self, lr):
        for variable in self.list_of_variables:
            variable.data -= (lr / np.sqrt(self.gradient_histories[variable])) * variable.grad
            self.gradient_histories[variable] += variable.grad**2

class AdamOptimizer:
    def __init__(self, list_of_layers, beta1=0.9, beta2=0.999, eps=0.00000001):
        self.list_of_layers = list_of_layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.list_of_variables = []
        for layer in self.list_of_layers:
            self.list_of_variables += layer.report_params()
        self.adam_mean = dict()
        self.adam_var = dict()
        for variable in self.list_of_variables:
            self.adam_mean[variable] = np.zeros(variable.shape)
            self.adam_var[variable] = np.zeros(variable.shape)
    def step(self, lr):
        for variable in self.list_of_variables:
            self.adam_mean[variable] = self.adam_mean[variable] * self.beta1 + variable.grad * (1 - self.beta1)
            self.adam_var[variable] = self.adam_var[variable] * self.beta2 + (variable.grad**2) * (1 - self.beta2)
            mean_hat = self.adam_mean[variable] / (1 - self.beta1)
            var_hat = self.adam_var[variable] / (1 - self.beta2)
            variable.data -= (lr * mean_hat)/(np.sqrt(var_hat) + self.eps)


class ReLUActivation(IdentityActivation):
    # During a forward pass, it just applies the sigmoid function
    def forward(self, x, evaluate=False):
        self.activation = x * (x > 0)
        return self.activation
    # During backprop, it passes the delta through its derivative
    def backprop(self, delta, y):
        return delta * (self.activation > 0)

class SoftplusActivation(IdentityActivation):
    # During a forward pass, it just applies the sigmoid function
    def forward(self, x, evaluate=False):
        self.activation = np.log(1.0 + np.exp(x))
        return self.activation
    # During backprop, it passes the delta through its derivative
    def backprop(self, delta, y):
        return delta * 1.0/(1.0 + np.exp(-self.activation))
    
class DropoutLayer(IdentityActivation):
    def __init__(self, predecessor, probability=0.5):
        self.predecessor = predecessor
        self.input_size = self.predecessor.output_size        
        self.output_size = self.input_size
        self.activation = np.zeros(self.output_size)
        self.gradient = np.zeros(self.output_size)
        self.init_weights()
        self.require_gradient = False
        # Noise
        self.probability = probability
    def forward(self, x, evaluate=False):
        if evaluate:
            self.activation = x
        else:
            dropout = np.random.choice([0, 1], size=x.shape, p=[self.probability, 1 - self.probability])
            self.activation = (x * dropout)/(1 - self.probability)
        return self.activation

class BatchNormLayer(IdentityActivation):
    def __init__(self, predecessor, eps=0.01):
        self.predecessor = predecessor
        self.input_size = self.predecessor.output_size
        self.output_size = self.input_size
        self.hidden = self.output_size[1]
        self.activation = np.zeros(self.output_size)
        self.gradient = np.zeros(self.output_size)
        self.init_params()
        self.zero_grad()
        self.require_gradient = True
        # Batchnorm requires a constant for "numerical stability"
        self.eps = eps
        # We need to save mean and variance as constants for backprop
        self.mean = np.zeros((1, self.hidden))
        self.var = np.zeros((1, self.hidden))
        # Also, save the xhat
        self.xhat = np.zeros((1, self.hidden))
        # We also want to keep running means and variances during training
        # These become evaluation statistics
        self.eval_mean = np.zeros((1, self.hidden))
        self.eval_var = np.zeros((1, self.hidden))
    def init_params(self):
        # Initialize gamma (mean) and beta (variance)
        self.gamma = Variable(np.ones((1, self.hidden)))
        self.beta = Variable(np.zeros((1, self.hidden)))
    def zero_grad(self):
        self.gamma.zero_grad()
        self.beta.zero_grad()
    def forward(self, x, evaluate=False):
        if evaluate:
            xhat = (x - self.eval_mean) / np.sqrt(self.eval_var + self.eps)
            self.activation = self.gamma.data * xhat + self.beta.data            
        else:
            # Batch mean and variance
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            # Evaluation mean and variance
            self.eval_mean = 0.9*self.eval_mean + 0.1*self.mean
            self.eval_var = 0.9*self.eval_var + 0.1*self.var
            # Calculate xhat and the final normalized activation
            self.xhat = (x - self.mean) / np.sqrt(self.var + self.eps)
            self.activation = self.gamma.data * self.xhat + self.beta.data
        return self.activation
    def backprop(self, delta, y):
        N = delta.shape[0]

        self.gamma.grad += np.sum(delta * self.xhat, axis=0)
        self.beta.grad += np.sum(delta, axis=0)

        x_mean = self.predecessor.activation - self.mean
        inv_var_eps = 1 / np.sqrt(self.var + self.eps)

        d_xhat = delta * self.gamma.data
        d_var = np.sum(d_xhat * x_mean, axis=0) * -0.5 * inv_var_eps**3
        d_mean = np.sum(d_xhat * -inv_var_eps, axis=0) + (d_var * np.mean(-2.0 * x_mean))
        delta = (d_xhat * inv_var_eps) + (d_var * 2 * x_mean / N) + (d_mean / N)

        return delta
    def report_params(self):
        return [self.beta, self.gamma]

def NLLCost(prediction, truth):
    return -np.mean(np.sum(truth*np.log(prediction) + (1.0-truth)*np.log(1.0-prediction), 1))

def CrossEntropy(prediction, truth):
    return -np.mean(np.sum(truth*np.log(prediction), 1))

def accuracy(prediction, labels):
    tests = prediction.argmax(axis=1) == labels
    return(tests.sum() / prediction.shape[0])
