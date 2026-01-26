import numpy as np

def softmaxXEntropy(x, y):
    """
    Compute softmax probabilities and cross-entropy loss.
    
    Parameters:
    -----------
    x : numpy array
        Logits from the model (num_classes, batch_size)
        Must be a np_tensor to store gradients
    y : numpy array
        True labels (batch_size, 1) or (batch_size,)
    
    Returns:
    --------
    preds : numpy array
        Predicted probabilities (num_classes, batch_size)
    cost : float
        Average cross-entropy loss
    """
    batch_size = x.shape[1]
    
    # Numerical stability: subtract max before exp
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    exp_scores = np.exp(x_shifted)
    probs = exp_scores / exp_scores.sum(axis=0, keepdims=True)
    preds = probs.copy()
    
    # Compute cost: -log(prob of correct class)
    y_hat = probs[y.squeeze(), np.arange(batch_size)]
    cost = np.sum(-np.log(y_hat + 1e-8)) / batch_size
    
    # Compute gradient: dL/dx = probs - one_hot(y)
    probs[y.squeeze(), np.arange(batch_size)] -= 1
    x.grad = probs.copy()
    
    return preds, cost
