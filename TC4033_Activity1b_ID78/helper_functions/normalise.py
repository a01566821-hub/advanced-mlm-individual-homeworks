# Normalization function
def normalise(x_mean, x_std, x_data):
    """
    Normalize data to have zero mean and unit standard deviation.
    
    Parameters:
    -----------
    x_mean : float
        Mean of training data
    x_std : float
        Standard deviation of training data
    x_data : numpy array
        Data to normalize
    
    Returns:
    --------
    normalized_data : numpy array
        Normalized data
    """
    return (x_data - x_mean) / x_std