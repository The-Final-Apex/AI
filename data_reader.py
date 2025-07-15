import pandas as pd

DATA = "mnist_train.csv"
DATA = pd.read_csv(DATA).to_numpy()

def extract_data(data, num_samples=1000):
    """
    Extracts a specified number of samples from the MNIST dataset.
    
    Parameters:
    - data: The MNIST dataset as a numpy array.
    - num_samples: The number of samples to extract.
    
    Returns:
    - A tuple containing the features and labels.
    """
    if num_samples > len(data):
        raise ValueError("num_samples exceeds the size of the dataset.")
    
    features = data[:num_samples, 1:] / 255.0  # Normalize pixel values
    labels = data[:num_samples, 0].astype(int)  # Convert labels to integers
    
    return features, labels

if __name__ == "__main__":
    features, labels = extract_data(DATA, num_samples=1000)
    print(f"Extracted {len(features)} samples.")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    
