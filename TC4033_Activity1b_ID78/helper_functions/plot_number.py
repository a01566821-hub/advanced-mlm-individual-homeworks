import matplotlib.pyplot as plt

def plot_number(image):
    """
    Plot an ASL sign image.

    Parameters:
    -----------
    image : numpy array
        Image array (28x28) to display
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()