from .create_minibatches import create_minibatches
from .np_tensor import np_tensor
from .softmax_xentropy import softmaxXEntropy
from .accuracy import accuracy


def train(model, epochs, mb_size=128, learning_rate=1e-3, x_train=None, y_train=None, x_val=None, y_val=None):
    """
    Train the neural network model.
    
    Parameters:
    -----------
    model : Sequential_layers
        Model to train
    epochs : int
        Number of training epochs
    mb_size : int
        Mini-batch size
    learning_rate : float
        Learning rate for gradient descent
    x_train : numpy array, optional
        Training input features. If None, tries to get from global namespace.
    y_train : numpy array, optional
        Training labels. If None, tries to get from global namespace.
    x_val : numpy array, optional
        Validation input features. If None, tries to get from global namespace.
    y_val : numpy array, optional
        Validation labels. If None, tries to get from global namespace.
    """
    import numpy as np
    import inspect
    
    # If parameters not provided, try to get from caller's global namespace
    if x_train is None or y_train is None or x_val is None or y_val is None:
        frame = inspect.currentframe().f_back
        globals_dict = frame.f_globals
        x_train = x_train if x_train is not None else globals_dict.get('x_train')
        y_train = y_train if y_train is not None else globals_dict.get('y_train')
        x_val = x_val if x_val is not None else globals_dict.get('x_val')
        y_val = y_val if y_val is not None else globals_dict.get('y_val')
        
        if x_train is None or y_train is None or x_val is None or y_val is None:
            raise ValueError("x_train, y_train, x_val, y_val must be provided as parameters or defined in the global namespace")
    
    # Reshape y_train if needed
    y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
    
    for epoch in range(epochs):
        # Iterate through mini-batches
        for x_batch, y_batch in create_minibatches(mb_size, x_train, y_train_reshaped):
            # Forward pass: transpose input to (784, batch_size)
            scores = model(x_batch.T.view(np_tensor))
            
            # Compute loss and gradients
            _, cost = softmaxXEntropy(scores, y_batch)
            
            # Backward pass: compute gradients
            model.backward()
            
            # Update parameters
            model.update(learning_rate)
        
        # Print progress after each epoch
        val_acc = accuracy(model, x_val, y_val, mb_size)
        print(f'Epoch {epoch+1}/{epochs} - Cost: {cost:.4f}, Validation Accuracy: {val_acc:.4f}')
    
    print(f'\nTraining completed!')
    print(f'Final validation accuracy: {accuracy(model, x_val, y_val, mb_size):.4f}')
