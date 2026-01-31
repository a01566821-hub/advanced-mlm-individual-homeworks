import numpy as np


class ReLU():
    """
    Rectified Linear Unit (ReLU) activation function.
    
    ReLU introduces non-linearity to the network, allowing it to learn
    complex patterns. It's defined as: f(x) = max(0, x)
    """
    def __call__(self, Z):
        """
        Forward pass: apply ReLU activation.
        
        Parameters:
        -----------
        Z : numpy array
            Input to the activation function
        
        Returns:
        --------
        A : numpy array
            Activated output (same shape as Z)
        """
        return np.maximum(0, Z)
    
    def backward(self, Z, A):
        """
        Backward pass: compute gradient of ReLU.
        
        Parameters:
        -----------
        Z : numpy array
            Input to ReLU (before activation)
        A : numpy array
            Output of ReLU (after activation)
            Must have A.grad attribute set from next layer
        
        The gradient is:
        - dL/dZ = dL/dA where Z > 0
        - dL/dZ = 0 where Z <= 0
        """
        Z.grad = A.grad.copy()
        Z.grad[Z <= 0] = 0
