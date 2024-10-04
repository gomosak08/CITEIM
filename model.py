import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import load_data
import json 
from functions import KeypointTransformer, train_model 
CUDA_LAUNCH_BLOCKING=1
# Replace 'your_file.json' with the path to your JSON file
with open('keypoints_dataset.json', 'r') as f:
    json_data = json.load(f)


# Assume you already have your model, data, and DataLoader ready
# Your model class


# Example of setting up the data and the model
# Assuming keypoints_data_loader is your DataLoader for keypoints data

batch_size = 12
n_frames = 60
n_keypoints = 75
n_features = 3
d_model = 128
num_classes = 29



# Initialize the model
model = KeypointTransformer(n_keypoints=n_keypoints, n_features=n_features, n_frames=n_frames, d_model=d_model, num_classes=num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training
num_epochs = 100

keypoints_data_loader = load_data(json_data)
# Assuming keypoints_data_loader is your DataLoader object
print(torch.cuda.is_available())
train_model(model, keypoints_data_loader, criterion, optimizer, num_epochs=num_epochs, device='cuda' if torch.cuda.is_available() else 'cpu')
print("training done")
MODEL_PATH = 'keypoint_transformer.pth'
torch.save(model.state_dict(), MODEL_PATH)

