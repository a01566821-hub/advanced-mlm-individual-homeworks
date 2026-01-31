import numpy as np

def split_val_test(x, y, pct=0.5, shuffle=True):
    '''
    Split the validation set into validation and test sets.
    
    Parameters:
    -----------
    x : numpy array
        Input features (validation set)
    y : numpy array
        Labels (validation set)
    pct : float, default=0.5
        Percentage of data to use for validation (rest goes to test)
    shuffle : bool, default=True
        Whether to shuffle the data before splitting
    
    Returns:
    --------
    x_val, y_val, x_test, y_test : numpy arrays
        Split validation and test sets
    '''
    assert x.shape[0] == y.shape[0], 'Error: x and y must have same number of samples'
    
    total_samples = x.shape[0]
    val_size = int(total_samples * pct)
    
    if shuffle:
        idxs = np.arange(total_samples)
        np.random.shuffle(idxs)
        x = x[idxs]
        y = y[idxs]
    
    # Split the data
    x_val = x[:val_size]
    y_val = y[:val_size]
    x_test = x[val_size:]
    y_test = y[val_size:]
    
    return x_val, y_val, x_test, y_test