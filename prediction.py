# Import necessary libraries and modules
import torch  # PyTorch library for tensor computations and model handling
import torch.nn as nn  # Neural network components from PyTorch
from functions import KeypointTransformer  # Custom model class for keypoint-based prediction

# Path to the saved model file
MODEL_PATH = 'keypoint_transformer.pth'


# Function to load the trained model
def load_model():
    """
    This function loads a pre-trained KeypointTransformer model with the given architecture.

    Returns:
        model (torch.nn.Module): The loaded KeypointTransformer model ready for inference.
    """

    # Define the model's architecture parameters (these should match the parameters used during training)
    n_keypoints = 75      # Number of keypoints (e.g., for pose estimation, this could be joints or landmarks)
    n_features = 3        # Number of features for each keypoint (x, y, and confidence score or similar)
    n_frames = 60         # Number of frames in the input sequence (how long the sequence of keypoints is)
    d_model = 128         # Dimensionality of the internal model's representation
    num_classes = 29      # Number of output classes (the number of classes the model can predict)

    # Create an instance of the KeypointTransformer model using the specified parameters
    model = KeypointTransformer(n_keypoints=n_keypoints, n_features=n_features, n_frames=n_frames, d_model=d_model, num_classes=num_classes)

    # Load the pre-trained model weights from the specified file path
    model.load_state_dict(torch.load(MODEL_PATH))

    # Set the model to evaluation mode, disabling dropout and other training-specific layers
    model.eval()

    return model  # Return the loaded model for inference


# Function to make predictions using the loaded model
def made_pred(keypoints, model):
    """
    This function makes predictions based on the keypoints input using the provided model.

    Args:
        keypoints (torch.Tensor): The input tensor containing keypoints data, typically of shape [batch_size, n_frames, n_keypoints, n_features].
        model (torch.nn.Module): The pre-trained model used for inference.

    Returns:
        predicted_classes (torch.Tensor): Tensor containing the predicted class indices for each input in the batch.
    """

    # Disable gradient calculations to improve performance during inference (we don't need to backpropagate)
    with torch.no_grad():
        # Forward pass: run the input keypoints through the model to obtain the raw class scores (logits)
        output = model(keypoints)

    # Use torch.argmax to select the class with the highest score for each input in the batch
    predicted_classes = torch.argmax(output, dim=1)

    return predicted_classes  # Return the predicted class indices
