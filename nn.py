import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.ndimage import zoom

#The basis
# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(hidden_size, input_size) * 0.01
        self.bias_hidden = np.zeros((hidden_size, 1))
        self.weights_hidden_output = np.random.randn(output_size, hidden_size) * 0.01
        self.bias_output = np.zeros((output_size, 1))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return z > 0
     
    def forward(self, X):
        self.z1 = np.dot(self.weights_input_hidden, X) + self.bias_hidden
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.weights_hidden_output, self.a1) + self.bias_output
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, Y, learning_rate):
        m = X.shape[1]
        dz2 = self.a2 - Y
        dw2 = (1 / m) * np.dot(dz2, self.a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        
        dz1 = np.dot(self.weights_hidden_output.T, dz2) * self.relu_derivative(self.z1)
        dw1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        
        self.weights_hidden_output -= learning_rate * dw2
        self.bias_output -= learning_rate * db2
        self.weights_input_hidden -= learning_rate * dw1
        self.bias_hidden -= learning_rate * db1
        

# Train the Neural Network
def train_model(nn, X_train, Y_train, epochs=50, learning_rate=0.1):
    losses = []
    for epoch in range(epochs):
        predictions = nn.forward(X_train)
        loss = -np.mean(Y_train * np.log(predictions + 1e-8))
        losses.append(loss)
        nn.backward(X_train, Y_train, learning_rate)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    # Plot Loss
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

# Predict a single example
def predict(nn, X):
    probabilities = nn.forward(X)
    prediction = np.argmax(probabilities, axis=0)
    return prediction[0]

# Draw a digit
def draw_digit():
    print("Draw your digit by clicking and dragging with the mouse.")
    canvas = np.zeros((280, 280))

    def on_draw(event):
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if 0 <= x + dx < 280 and 0 <= y + dy < 280:
                        canvas[y + dy, x + dx] = 255

            ax.imshow(canvas, cmap="gray")
            fig.canvas.draw()

    fig, ax = plt.subplots()
    ax.imshow(canvas, cmap="gray")
    fig.canvas.mpl_connect("motion_notify_event", on_draw)
    plt.show()

    # Resize to 28x28
    small_canvas = zoom(canvas, (28 / 280, 28 / 280))
    return small_canvas.flatten() / 255.0

# Main Program
if __name__ == "__main__":
    # Load MNIST Dataset
    mnist = fetch_openml("mnist_784", version=1)
    data = mnist.data / 255.0
    labels = mnist.target.astype("int")
    
    # One-hot encode labels
    labels_one_hot = np.eye(10)[labels]
    
    # Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, labels_one_hot, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.T, Y_test.T
    
    # Initialize Neural Network
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    
    # Train Neural Network
    print("Training the neural network...")
    train_model(nn, X_train, Y_train, epochs=50, learning_rate=0.1)
    
    # Evaluate
    print("You can now draw a digit.")
    user_digit = draw_digit()
    user_digit = user_digit.reshape(784, 1)  # Reshape to match input size
    prediction = predict(nn, user_digit)
    print(f"The model predicts: {prediction}")
