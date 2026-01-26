import numpy as np
from .create_minibatches import create_minibatches
from .np_tensor import np_tensor

def accuracy(model, x, y, mb_size):
    """
    Compute accuracy of the model on a dataset.
    
    Parameters:
    -----------
    model : Sequential_layers
        Trained model
    x : numpy array
        Input data (num_samples, features)
    y : numpy array
        True labels (num_samples, 1) or (num_samples,)
    mb_size : int
        Mini-batch size for evaluation
    
    Returns:
    --------
    float
        Accuracy (between 0 and 1)
    """
    correct = 0
    total = 0
    
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    for x_batch, y_batch in create_minibatches(mb_size, x, y, shuffle=False):
        pred = model(x_batch.T.view(np_tensor))
        pred_classes = np.argmax(pred, axis=0)
        correct += np.sum(pred_classes == y_batch.squeeze())
        total += pred.shape[1]
    
    return correct / total
