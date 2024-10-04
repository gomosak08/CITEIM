# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim

# Define the KeypointTransformer model class
class KeypointTransformer(nn.Module):
    def __init__(self, n_keypoints=75, n_features=3, n_frames=60, d_model=128, nhead=8, num_classes=2, num_layers=6):
        """
        Initializes the KeypointTransformer model for sequence-based classification tasks.

        Args:
            n_keypoints (int): Number of keypoints (e.g., 75 for pose + hand landmarks).
            n_features (int): Number of features per keypoint (e.g., x, y, z coordinates).
            n_frames (int): Number of frames in the input sequence.
            d_model (int): Dimension of the model's feature space.
            nhead (int): Number of attention heads in the transformer.
            num_classes (int): Number of output classes.
            num_layers (int): Number of transformer encoder layers.
        """
        super(KeypointTransformer, self).__init__()
        
        self.n_keypoints = n_keypoints
        self.n_frames = n_frames
        self.d_model = d_model
        
        # Linear layer to project keypoint features to the transformer input dimension
        self.input_projection = nn.Linear(n_keypoints * n_features, d_model)
        
        # Positional encoding for the keypoints (trainable)
        self.positional_encoding = nn.Parameter(torch.zeros(n_frames, d_model))
        
        # Encoder layer and transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_frames, n_keypoints, n_features).
        
        Returns:
            torch.Tensor: Output logits for each class.
        """
        # x shape: (batch_size, n_frames, n_keypoints, n_features)
        batch_size = x.shape[0]
        
        # Reshape keypoints to (batch_size, n_frames, n_keypoints * n_features)
        x = x.view(batch_size, self.n_frames, -1)
        
        # Project the keypoint features to match d_model size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Reshape to (n_frames, batch_size, d_model) as expected by nn.Transformer
        x = x.permute(1, 0, 2)  # (n_frames, batch_size, d_model)
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output for the final frame
        x = x[-1]  # shape: (batch_size, d_model)
        
        # Classification head
        x = self.fc(x)
        return x


# Example function for calculating accuracy
def accuracy_fn(predictions, labels):
    """
    Computes the accuracy for a batch of predictions.

    Args:
        predictions (torch.Tensor): Model outputs (logits).
        labels (torch.Tensor): True labels.
    
    Returns:
        float: Accuracy for the batch.
    """
    _, preds = torch.max(predictions, 1)  # Get the index of the highest logit
    return torch.sum(preds == labels).item() / len(labels)


# Training loop function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Trains the KeypointTransformer model using the provided dataloader.

    Args:
        model (KeypointTransformer): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        num_epochs (int): Number of training epochs.
        device (str): Device to perform training on ('cuda' or 'cpu').

    Returns:
        None
    """
    model = model.to(device)  # Move model to the selected device
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_accuracy = 0.0

        for batch_idx, (keypoints, labels) in enumerate(dataloader):
            # Move data to the device
            keypoints, labels = keypoints.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(keypoints)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            running_loss += loss.item()
            running_accuracy += accuracy_fn(outputs, labels)
        
        # Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_accuracy / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("Training complete.")
