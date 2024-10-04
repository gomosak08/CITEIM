import torch
import torch.nn as nn
import torch.optim as optim

class KeypointTransformer(nn.Module):
    def __init__(self, n_keypoints=75, n_features=3, n_frames=60, d_model=128, nhead=8, num_classes=2, num_layers=6):
        super(KeypointTransformer, self).__init__()
        
        self.n_keypoints = n_keypoints
        self.n_frames = n_frames
        self.d_model = d_model
        
        # Linear layer to project keypoint features to the transformer input dimension
        self.input_projection = nn.Linear(n_keypoints * n_features, d_model)
        
        # Positional encoding for the keypoints
        self.positional_encoding = nn.Parameter(torch.zeros(n_frames, d_model))
        
        # Encoder layer and transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
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
        
        # Take the output for the final frame (can also use pooling or attention)
        x = x[-1]  # shape: (batch_size, d_model)
        
        # Classification head
        x = self.fc(x)
        return x

# Example function for calculating accuracy
def accuracy_fn(predictions, labels):
    _, preds = torch.max(predictions, 1)
    return torch.sum(preds == labels).item() / len(labels)

# Training loop function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
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
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_accuracy / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    print("Training complete.")