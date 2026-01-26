import copy
import numpy as np
from .sequential_layers import Sequential_layers
from .train import train
from .accuracy import accuracy

def train_and_eval_model(config, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Train a single model configuration and evaluate it.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - 'layers': list of layer objects
        - 'mb_size': mini-batch size
        - 'learning_rate': learning rate
        - 'epochs': number of epochs
    x_train, y_train : numpy arrays
        Training data
    x_val, y_val : numpy arrays
        Validation data
    x_test, y_test : numpy arrays
        Test data
    
    Returns:
    --------
    dict
        Dictionary with model, config, and test_accuracy
    """
    # Create a fresh model with the configuration
    model = Sequential_layers(copy.deepcopy(config['layers']))
    mb_size = config['mb_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    
    # Train the model
    train(model, epochs, mb_size, learning_rate, x_train, y_train, x_val, y_val)
    
    # Evaluate on test set
    test_acc = accuracy(model, x_test, y_test, mb_size)
    
    return {
        'model': model,
        'config': config,
        'test_accuracy': test_acc
    }

def parallel_train_models(model_configs, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Train multiple model configurations and return the best one.
    
    Parameters:
    -----------
    model_configs : list of dict
        List of configuration dictionaries, each with:
        - 'layers': list of layer objects
        - 'mb_size': mini-batch size
        - 'learning_rate': learning rate
        - 'epochs': number of epochs
    x_train, y_train : numpy arrays
        Training data
    x_val, y_val : numpy arrays
        Validation data
    x_test, y_test : numpy arrays
        Test data
    
    Returns:
    --------
    dict
        Dictionary with:
        - 'results': list of all results
        - 'best_result': result with highest test accuracy
    """
    print("Starting parallel training for multiple configurations...\n")
    
    results = []
    best_result = None
    best_accuracy = 0.0
    
    for i, config in enumerate(model_configs):
        print(f"Training configuration {i+1}/{len(model_configs)}...")
        
        try:
            result = train_and_eval_model(
                config, x_train, y_train, x_val, y_val, x_test, y_test
            )
            
            results.append(result)
            
            # Track best model
            if result['test_accuracy'] > best_accuracy:
                best_accuracy = result['test_accuracy']
                best_result = result
            
            # Print configuration summary
            print(f"\nConfiguration:")
            layer_names = [type(layer).__name__ for layer in config['layers']]
            print(f"  Layers: {layer_names}")
            print(f"  Mini-batch size: {config['mb_size']}")
            print(f"  Learning rate: {config['learning_rate']}")
            print(f"  Epochs: {config['epochs']}")
            print(f"  Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)\n")
            
        except Exception as e:
            print(f"Error training configuration {i+1}: {e}\n")
            continue
    
    # Print best model summary
    if best_result:
        print(f"\nBest Model Configuration Found:")
        layer_names = [type(layer).__name__ for layer in best_result['config']['layers']]
        print(f"  Layers: {layer_names}")
        print(f"  Mini-batch size: {best_result['config']['mb_size']}")
        print(f"  Learning rate: {best_result['config']['learning_rate']}")
        print(f"  Epochs: {best_result['config']['epochs']}")
        print(f"  Best Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)\n")
    
    return {
        'results': results,
        'best_result': best_result
    }
