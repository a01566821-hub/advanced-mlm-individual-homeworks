import numpy as np

def create_minibatches(mb_size, x, y, shuffle=True):
    """
    Create mini-batches from the dataset.
    
    Parameters:
    -----------
    mb_size : int
        Size of each mini-batch
    x : numpy array
        Input data (num_samples, features)
    y : numpy array
        Labels (num_samples, 1) or (num_samples,)
    shuffle : bool
        Whether to shuffle the data before creating batches
    
    Returns:
    --------
    generator
        Generator yielding (x_batch, y_batch) tuples
    """
    assert x.shape[0] == y.shape[0], 'Error en cantidad de muestras'
    total_data = x.shape[0]
    
    if shuffle:
        idxs = np.arange(total_data)
        np.random.shuffle(idxs)
        x = x[idxs]
        y = y[idxs]
    
    # Reshape y to (num_samples, 1) for consistency
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    return ((x[i:i+mb_size], y[i:i+mb_size]) for i in range(0, total_data, mb_size))
