import numpy as np
from .np_tensor import np_tensor


class Linear():
    """
    Fully connected linear layer.
    
    Performs the transformation: Z = WX + b
    where W is the weight matrix and b is the bias vector.
    """
    def __init__(self, input_size, output_size):
        """
        Initialize the linear layer.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output features
        
        Uses Kaiming He initialization for weights (suitable for ReLU activation).
        """
        # Kaiming He initialization: divide by sqrt(input_size/2)
        # This helps with gradient flow when using ReLU activation
        self.W = (np.random.randn(output_size, input_size) / np.sqrt(input_size/2)).view(np_tensor)
        self.b = (np.zeros((output_size, 1))).view(np_tensor)
    
    def __call__(self, X):
        """
        Forward pass through the linear layer.
        
        Parameters:
        -----------
        X : numpy array
            Input data (input_size, batch_size)
        
        Returns:
        --------
        Z : numpy array
            Output (output_size, batch_size)
        """
        Z = self.W @ X + self.b
        return Z
    
    def backward(self, X, Z):
        """
        Backward pass: compute gradients.
        
        Parameters:
        -----------
        X : numpy array
            Input to the layer (input_size, batch_size)
        Z : numpy array
            Output of the layer (output_size, batch_size)
            Must have Z.grad attribute set from next layer
        """
        # Gradient w.r.t. input X: dL/dX = W^T @ dL/dZ
        X.grad = self.W.T @ Z.grad
        
        # Gradient w.r.t. weights W: dL/dW = dL/dZ @ X^T
        self.W.grad = Z.grad @ X.T
        
        # Gradient w.r.t. bias b: dL/db = sum(dL/dZ, axis=1)
        self.b.grad = np.sum(Z.grad, axis=1, keepdims=True)
