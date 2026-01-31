import numpy as np


def softmaxXEntropy(x, y):
    """
    Compute softmax probabilities and cross-entropy loss.
    
    Parameters:
    -----------
    x : numpy array
        Raw scores from the network (num_classes, batch_size)
    y : numpy array
        True labels (batch_size, 1)
    
    Returns:
    --------
    preds : numpy array
        Predicted probabilities (num_classes, batch_size)
    cost : float
        Average cross-entropy loss
    """
    batch_size = x.shape[1]
    
    # Compute softmax: exp(x) / sum(exp(x))
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    exp_scores = np.exp(x_shifted)
    probs = exp_scores / exp_scores.sum(axis=0, keepdims=True)
    preds = probs.copy()
    
    # Compute cross-entropy loss: -log(p_true_class)
    # Get probability of true class for each sample
    y_hat = probs[y.squeeze(), np.arange(batch_size)]
    cost = np.sum(-np.log(y_hat + 1e-8)) / batch_size  # Add small epsilon for numerical stability
    
    # Compute gradient: dL/dx = probs - one_hot(y)
    # For the true class, subtract 1; others remain as is
    probs[y.squeeze(), np.arange(batch_size)] -= 1
    x.grad = probs.copy()
    
    return preds, cost
