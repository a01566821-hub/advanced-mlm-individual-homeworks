import numpy as np
from .relu import ReLU


class Sequential_layers():
    """
    Sequential container for neural network layers.
    
    Manages forward pass, backward pass, and parameter updates
    for a sequence of layers.
    """
    def __init__(self, layers):
        """
        Initialize the sequential model.
        
        Parameters:
        -----------
        layers : list
            List of layer objects (Linear, ReLU, etc.)
        """
        self.layers = layers
        self.x = None
        self.outputs = {}
    
    def __call__(self, X):
        """
        Forward pass through all layers.
        
        Parameters:
        -----------
        X : numpy array
            Input data (input_size, batch_size)
        
        Returns:
        --------
        output : numpy array
            Final output after passing through all layers
        """
        self.x = X
        self.outputs['l0'] = self.x
        
        # Pass through each layer sequentially
        for i, layer in enumerate(self.layers, 1):
            self.x = layer(self.x)
            self.outputs['l' + str(i)] = self.x
        
        return self.x
    
    def backward(self):
        """
        Backward pass through all layers (backpropagation).
        
        Computes gradients for all parameters by propagating
        the gradient backwards through the network.
        """
        # Process layers in reverse order
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # Get input and output for this layer
            layer_input = self.outputs['l' + str(i)]
            layer_output = self.outputs['l' + str(i + 1)]
            
            # Call backward method of the layer
            layer.backward(layer_input, layer_output)
    
    def update(self, learning_rate=1e-3):
        """
        Update parameters using gradient descent.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent update
        """
        for layer in self.layers:
            # Skip activation layers (they don't have parameters)
            if isinstance(layer, ReLU):
                continue
            
            # Update weights and biases: θ = θ - α * ∇θ
            layer.W = layer.W - learning_rate * layer.W.grad
            layer.b = layer.b - learning_rate * layer.b.grad
    
    def predict(self, X):
        """
        Make a prediction for a single sample.
        
        Parameters:
        -----------
        X : numpy array
            Input sample (input_size, 1)
        
        Returns:
        --------
        prediction : int
            Predicted class index
        """
        return np.argmax(self.__call__(X))
