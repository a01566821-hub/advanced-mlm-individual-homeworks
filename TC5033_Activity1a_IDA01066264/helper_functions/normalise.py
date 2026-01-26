import numpy as np

def normalise(x_mean, x_std, x_data):
    """
    Normalize data to have zero mean and unit variance.
    
    Parameters:
    -----------
    x_mean : float
        Mean value to subtract
    x_std : float
        Standard deviation to divide by
    x_data : numpy array
        Data to normalize
    
    Returns:
    --------
    numpy array
        Normalized data
    """
    return (x_data - x_mean) / x_std
