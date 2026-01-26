import matplotlib.pyplot as plt

def plot_number(image):
    """
    Plot a single image.
    
    Parameters:
    -----------
    image : numpy array
        Image to plot (should be 28x28 for MNIST)
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image.squeeze(), cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()
