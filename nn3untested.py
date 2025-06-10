import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.ndimage import zoom, rotate
import time
from IPython.display import clear_output
import seaborn as sns
from matplotlib.widgets import Button
import warnings

# Ignor warnings for cleaner output
warnings.filterwarnings('ignore')

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, 
                 activation='leaky_relu', initialization='he', 
                 dropout_rate=0.2, batch_norm=True):
        """
        Enhanced neural network with multiple hidden layers and advanced features
        
        Parameters:
        - input_size: Dimension of input features
        - hidden_sizes: List of integers specifying hidden layer sizes
        - output_size: Number of output classes
        - activation: Activation function ('leaky_relu', 'elu', 'swish')
        - initialization: Weight initialization method ('he', 'xavier')
        - dropout_rate: Dropout rate for regularization (0 = no dropout)
        - batch_norm: Whether to use batch normalization
        """
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation_type = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.initialize_weights(initialization)
        self.initialize_optimizer_params()
        
    def initialize_weights(self, initialization):
        """Advanced weight initialization with layer-specific scaling"""
        self.weights = []
        self.biases = []
        self.gammas = []  
        self.betas = []
        # The last too are for batch normalisation
        
        for i in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i+1]
            
            if initialization == 'he':
                scale = np.sqrt(2.0 / fan_in)
            elif initialization == 'xavier':
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            else:
                scale = 0.01
                
            self.weights.append(np.random.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros((fan_out, 1)))
            
            if self.batch_norm and i < len(self.layer_sizes)-2:
                self.gammas.append(np.ones((fan_out, 1)))
                self.betas.append(np.zeros((fan_out, 1)))
    
    def initialize_optimizer_params(self):
        """Initialize Adam optimizer parameters for all weights and biases"""
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        
        if self.batch_norm:
            self.m_gamma = [np.zeros_like(g) for g in self.gammas]
            self.v_gamma = [np.zeros_like(g) for g in self.gammas]
            self.m_beta = [np.zeros_like(b) for b in self.betas]
            self.v_beta = [np.zeros_like(b) for b in self.betas]
    
    def activation(self, z, derivative=False):
        """Enhanced activation functions with multiple options"""
        if self.activation_type == 'leaky_relu':
            alpha = 0.01
            if derivative:
                return (z > 0) + alpha * (z <= 0)
            return np.maximum(alpha * z, z)
            
        elif self.activation_type == 'elu':
            alpha = 1.0
            if derivative:
                return (z > 0) + (z <= 0) * (alpha * np.exp(z))
            return np.where(z > 0, z, alpha * (np.exp(z) - 1))
            
        elif self.activation_type == 'swish':
            beta = 1.0
            sigmoid = 1 / (1 + np.exp(-beta * z))
            if derivative:
                return beta * sigmoid + z * beta * sigmoid * (1 - sigmoid)
            return z * sigmoid
            
        else:  # default to ReLU
            if derivative:
                return (z > 0).astype(float)
            return np.maximum(0, z)
    
    def softmax(self, z):
        """Numerically stable softmax"""
        shift_z = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def batch_norm_forward(self, z, gamma, beta, layer_idx, mode='train'):
        """Batch normalization forward pass"""
        if mode == 'train':
            self.batch_mean[layer_idx] = np.mean(z, axis=1, keepdims=True)
            self.batch_var[layer_idx] = np.var(z, axis=1, keepdims=True)
            
            # update running mean and inference
            # im going to abbreviate inference as infer for now
            if not hasattr(self, 'running_mean'):
                self.running_mean = [np.zeros_like(m) for m in self.batch_mean]
                self.running_var = [np.zeros_like(v) for v in self.batch_var]
                
            self.running_mean[layer_idx] = 0.9 * self.running_mean[layer_idx] + 0.1 * self.batch_mean[layer_idx]
            self.running_var[layer_idx] = 0.9 * self.running_var[layer_idx] + 0.1 * self.batch_var[layer_idx]
            
            z_norm = (z - self.batch_mean[layer_idx]) / np.sqrt(self.batch_var[layer_idx] + 1e-8)
            return gamma * z_norm + beta
            
        else:  # infer mode
            z_norm = (z - self.running_mean[layer_idx]) / np.sqrt(self.running_var[layer_idx] + 1e-8)
            return gamma * z_norm + beta
    
    def dropout(self, a, mode='train'):
        """Dropout regularization"""
        if mode == 'train' and self.dropout_rate > 0:
            self.mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            return a * self.mask
        return a
    
    def forward(self, X, mode='train'):
        """Forward pass with optional batch norm and dropout"""
        self.caches = []
        self.batch_mean = []
        self.batch_var = []
        a = X
        
        for i in range(len(self.weights)-1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            
            if self.batch_norm:
                z = self.batch_norm_forward(z, self.gammas[i], self.betas[i], i, mode)
                self.batch_mean.append(np.mean(z, axis=1, keepdims=True))
                self.batch_var.append(np.var(z, axis=1, keepdims=True))
            
            a = self.activation(z)
            a = self.dropout(a, mode)
            
            self.caches.append((z, a))
        
        # The output layer (no activation, dropout, or batch norm)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = self.softmax(z)
        self.caches.append((z, a))
        
        return a
    
    def backward(self, X, Y, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Backward pass with Adam optimization"""
        m = X.shape[1]
        grads = {}
        L = len(self.weights)
        
        # output layr gradient
        dz = self.caches[-1][1] - Y
        grads[f'dW{L-1}'] = (1 / m) * np.dot(dz, self.caches[-2][1].T)
        grads[f'db{L-1}'] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        da_prev = np.dot(self.weights[-1].T, dz)
        
        # Layers in hiding 👀
        for l in reversed(range(L-1)):
            z, a = self.caches[l]
            a_prev = X if l == 0 else self.caches[l-1][1]
            
            dz = da_prev * self.activation(z, derivative=True)
            
            if self.batch_norm and l < L-1:
                # batch norm backprop
                z_norm = (z - self.batch_mean[l]) / np.sqrt(self.batch_var[l] + 1e-8)
                dgamma = np.sum(dz * z_norm, axis=1, keepdims=True) / m
                dbeta = np.sum(dz, axis=1, keepdims=True) / m
                dz = dz * self.gammas[l] / np.sqrt(self.batch_var[l] + 1e-8)
                
                # update for gamma and beta (Adam)
                self.m_gamma[l] = beta1 * self.m_gamma[l] + (1 - beta1) * dgamma
                self.v_gamma[l] = beta2 * self.v_gamma[l] + (1 - beta2) * (dgamma**2)
                m_gamma_corr = self.m_gamma[l] / (1 - beta1**t)
                v_gamma_corr = self.v_gamma[l] / (1 - beta2**t)
                self.gammas[l] -= learning_rate * m_gamma_corr / (np.sqrt(v_gamma_corr) + epsilon)
                
                self.m_beta[l] = beta1 * self.m_beta[l] + (1 - beta1) * dbeta
                self.v_beta[l] = beta2 * self.v_beta[l] + (1 - beta2) * (dbeta**2)
                m_beta_corr = self.m_beta[l] / (1 - beta1**t)
                v_beta_corr = self.v_beta[l] / (1 - beta2**t)
                self.betas[l] -= learning_rate * m_beta_corr / (np.sqrt(v_beta_corr) + epsilon)
            
            grads[f'dW{l}'] = (1 / m) * np.dot(dz, a_prev.T)
            grads[f'db{l}'] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            
            if l > 0:
                da_prev = np.dot(self.weights[l].T, dz)
        
        # update for weights and biases (Adam)
        for l in range(L):
            # weight update
            self.m[l] = beta1 * self.m[l] + (1 - beta1) * grads[f'dW{l}']
            self.v[l] = beta2 * self.v[l] + (1 - beta2) * (grads[f'dW{l}']**2)
            m_corr = self.m[l] / (1 - beta1**t)
            v_corr = self.v[l] / (1 - beta2**t)
            self.weights[l] -= learning_rate * m_corr / (np.sqrt(v_corr) + epsilon)
            
            # blases update
            self.m_b[l] = beta1 * self.m_b[l] + (1 - beta1) * grads[f'db{l}']
            self.v_b[l] = beta2 * self.v_b[l] + (1 - beta2) * (grads[f'db{l}']**2)
            m_b_corr = self.m_b[l] / (1 - beta1**t)
            v_b_corr = self.v_b[l] / (1 - beta2**t)
            self.biases[l] -= learning_rate * m_b_corr / (np.sqrt(v_b_corr) + epsilon)
    
    def compute_loss(self, Y_hat, Y, l2_lambda=0.001):
        """Compute cross-entropy loss with L2 regularization"""
        m = Y.shape[1]
        cross_entropy = -np.mean(Y * np.log(Y_hat + 1e-8))
        
        # L2 regularization
        l2_penalty = 0
        for w in self.weights:
            l2_penalty += np.sum(w**2)
        l2_penalty = (l2_lambda / (2 * m)) * l2_penalty
        
        return cross_entropy + l2_penalty
    
    def predict(self, X):
        """Make predictions"""
        probs = self.forward(X, mode='test')
        return np.argmax(probs, axis=0)
    
    def evaluate(self, X, Y):
        """Evaluate model performance"""
        Y_hat = self.forward(X, mode='test')
        predictions = np.argmax(Y_hat, axis=0)
        labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == labels)
        return accuracy

def train_model(nn, X_train, Y_train, X_val, Y_val, 
                epochs=100, batch_size=128, learning_rate=0.001,
                lr_decay=0.95, early_stopping_patience=5):
    """
    The new training procedure now has:
    - mini-batch training
    - learning rate decay
    - early stopping
    - training/validation metrics tracking
    - visual feedback
    - feel free to put this in md
    """
    m = X_train.shape[1]
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Convert data to column vectors if needed
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
    if len(Y_train.shape) == 1:
        Y_train = Y_train.reshape(-1, 1)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Shuffle training data
        permutation = np.random.permutation(m)
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]
        
        # mini batch training
        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[:, i:i+batch_size]
            Y_batch = Y_train_shuffled[:, i:i+batch_size]
            
            # Forward and backward pass
            nn.forward(X_batch)
            nn.backward(X_batch, Y_batch, learning_rate, epoch+1)
        
        # compute metrics
        train_pred = nn.forward(X_train, mode='test')
        train_loss = nn.compute_loss(train_pred, Y_train)
        train_acc = nn.evaluate(X_train, Y_train)
        
        val_pred = nn.forward(X_val, mode='test')
        val_loss = nn.compute_loss(val_pred, Y_val)
        val_acc = nn.evaluate(X_val, Y_val)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate decay
        learning_rate *= lr_decay
        
        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model (simplified)
            best_weights = [w.copy() for w in nn.weights]
            best_biases = [b.copy() for b in nn.biases]
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                # Restore best model
                nn.weights = [w.copy() for w in best_weights]
                nn.biases = [b.copy() for b in best_biases]
                break
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc*100:.2f}% - Val Acc: {val_acc*100:.2f}%")
        
        # Plot progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            clear_output(wait=True)
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')
            
            plt.tight_layout()
            plt.show()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

class DigitCanvas:
    """Interactive digit drawing canvas with enhanced features"""
    def __init__(self):
        self.canvas = np.zeros((280, 280))
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Draw a digit (0-9) and click 'Recognize'")
        self.ax.imshow(self.canvas, cmap='gray_r', vmin=0, vmax=1)
        
        # Add recognize button
        self.button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, 'Recognize')
        self.button.on_clicked(self.recognize)
        
        # Add clear button
        self.clear_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.clear_button = Button(self.clear_ax, 'Clear')
        self.clear_button.on_clicked(self.clear_canvas)
        
        # Connect drawing events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_draw)
        self.fig.canvas.mpl_connect('button_press_event', self.on_draw)
        
        self.prediction = None
    
    def on_draw(self, event):
        """Handle drawing on canvas"""
        if event.inaxes != self.ax or (event.button != 1 and not hasattr(event, 'dblclick')):
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Draw a thicker point
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= 10 and 0 <= x + dx < 280 and 0 <= y + dy < 280:
                    # Smooth intensity based on distance
                    intensity = max(0, min(1, (10 - dist) / 10 + 0.3))
                    self.canvas[y + dy, x + dx] = min(1, self.canvas[y + dy, x + dx] + intensity)
        
        self.ax.imshow(self.canvas, cmap='gray_r', vmin=0, vmax=1)
        self.fig.canvas.draw()
    
    def clear_canvas(self, event):
        """Clear the drawing canvas"""
        self.canvas.fill(0)
        self.ax.imshow(self.canvas, cmap='gray_r', vmin=0, vmax=1)
        self.fig.canvas.draw()
        if hasattr(self, 'pred_text'):
            self.pred_text.remove()
            del self.pred_text
    
    def recognize(self, event):
        """Process the drawn digit and make prediction"""
        # Preprocess the image
        small_canvas = zoom(self.canvas, (28/280, 28/280))
        
        # Center the digit
        small_canvas = self.center_digit(small_canvas)
        
        # Normalize and flatten
        digit = small_canvas.flatten().reshape(-1, 1)
        
        # Make prediction
        if hasattr(self, 'model'):
            probabilities = self.model.forward(digit, mode='test')
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Display prediction
            if hasattr(self, 'pred_text'):
                self.pred_text.remove()
                
            self.pred_text = self.ax.text(0.5, -0.1, 
                                        f"Prediction: {prediction} (Confidence: {confidence*100:.1f}%)",
                                        transform=self.ax.transAxes,
                                        ha='center', fontsize=12, color='red')
            self.fig.canvas.draw()
            
            return prediction
        else:
            print("Model not loaded!")
            return None
    
    def center_digit(self, image, threshold=0.1):
        """Center the digit in the image"""
        # Binarize the image
        binary = image > threshold
        
        # Find the bounding box
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, 27)
        cmin, cmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, 27)
        
        # Calculate center of mass
        y_indices, x_indices = np.indices(image.shape)
        y_center = np.sum(y_indices * image) / np.sum(image) if np.sum(image) > 0 else 14
        x_center = np.sum(x_indices * image) / np.sum(image) if np.sum(image) > 0 else 14
        
        # Calculate required shift
        y_shift = int(14 - y_center)
        x_shift = int(14 - x_center)
        
        # Apply shift
        centered = np.roll(image, y_shift, axis=0)
        centered = np.roll(centered, x_shift, axis=1)
        
        return centered

def load_and_preprocess_data():
    """Load and preprocess MNIST data with enhancements"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype('int')
    
    # Normalize and reshape
    X = X / 255.0
    X = X.reshape(-1, 784).T  # Transpose to (784, n_samples)
    
    # One-hot encode labels
    y_one_hot = np.eye(10)[y]
    y_one_hot = y_one_hot.T  # Transpose to (10, n_samples)
    
    # Split into train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.T, y_one_hot.T, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y[y.reshape(-1)[X_temp.shape[0]:]]
    )
    
    # Transpose back to (features, samples)
    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
    y_train, y_val, y_test = y_train.T, y_val.T, y_test.T
    
    # Data augmentation - add rotated versions (simplified)
    def rotate_images(images, angle_range=(-15, 15)):
        augmented = []
        for img in images.T:
            angle = np.random.uniform(angle_range[0], angle_range[1])
            rotated = rotate(img.reshape(28, 28), angle, reshape=False, mode='nearest')
            augmented.append(rotated.flatten())
        return np.array(augmented).T
    
    # Small rotation augmentation
    X_train_aug = rotate_images(X_train, angle_range=(-10, 10))
    X_train = np.hstack((X_train, X_train_aug))
    y_train = np.hstack((y_train, y_train))
    
    print(f"Training set: {X_train.shape[1]} samples")
    print(f"Validation set: {X_val.shape[1]} samples")
    print(f"Test set: {X_test.shape[1]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(nn, X_test, y_test):
    """Comprehensive model evaluation"""
    print("\nEvaluating model on test set...")
    
    # Predictions
    y_pred = nn.predict(X_test)
    y_true = np.argmax(y_test, axis=0)
    
    # Accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Visual inspection of some examples
    print("\nVisual inspection of predictions:")
    plt.figure(figsize=(12, 6))
    indices = np.random.choice(X_test.shape[1], 12, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i+1)
        plt.imshow(X_test[:, idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main program execution"""
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    # Initialize neural network
    print("\nInitializing neural network...")
    nn = NeuralNetwork(input_size=784, 
                      hidden_sizes=[256, 128, 64], 
                      output_size=10,
                      activation='elu',
                      initialization='he',
                      dropout_rate=0.3,
                      batch_norm=True)
    
    # Train the model
    print("\nTraining model...")
    train_losses, val_losses, train_acc, val_acc = train_model(
        nn, X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=256,
        learning_rate=0.001,
        lr_decay=0.98,
        early_stopping_patience=10
    )
    
    # Evaluate on test set
    evaluate_model(nn, X_test, y_test)
    
    # Interactive digit recognition
    print("\nLaunching interactive digit recognition...")
    canvas = DigitCanvas()
    canvas.model = nn  # Pass the trained model to the canvas
    plt.show()

if __name__ == "__main__":
    main()
