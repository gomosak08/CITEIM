# Import necessary libraries and modules
import torch  # For tensor computations and model handling
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimization algorithms
from dataloader import load_data  # Custom function to load data into DataLoader
import json  # For handling JSON data files
from functions import KeypointTransformer, train_model  # Custom model and training function

# Set environment variable to prevent CUDA errors (for debugging purposes)
CUDA_LAUNCH_BLOCKING = 1

# Load keypoints dataset from JSON file
with open('keypoints_dataset.json', 'r') as f:
    json_data = json.load(f)  # Load the dataset in JSON format

# Define hyperparameters and model configurations
batch_size = 12  # Number of samples per batch
n_frames = 60  # Number of frames per input sequence
n_keypoints = 75  # Number of keypoints (e.g., 33 for pose and 21 each for left and right hands)
n_features = 3  # Number of features per keypoint (x, y, z)
d_model = 128  # Model dimension for transformer
num_classes = 29  # Number of output classes for classification

# Initialize the KeypointTransformer model with the specified parameters
model = KeypointTransformer(n_keypoints=n_keypoints, n_features=n_features, n_frames=n_frames, d_model=d_model, num_classes=num_classes)

# Define the loss function (cross-entropy for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer with a learning rate of 0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the data using a DataLoader (assuming load_data returns a DataLoader)
keypoints_data_loader = load_data(json_data)

# Check if CUDA (GPU) is available and print result
print(torch.cuda.is_available())

# Train the model using the specified parameters
num_epochs = 100  # Number of training epochs
train_model(model, keypoints_data_loader, criterion, optimizer, num_epochs=num_epochs, device='cuda' if torch.cuda.is_available() else 'cpu')

print("Training complete")

# Save the trained model weights to a file for later use
MODEL_PATH = 'keypoint_transformer.pth'
torch.save(model.state_dict(), MODEL_PATH)  # Save model weights to the specified file path
