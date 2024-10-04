import torch
import torch.nn as nn
from functions import KeypointTransformer
import json

def load_model():
    # Instantiate the model (ensure the model structure is identical to when you trained it)
    n_keypoints = 75
    n_features = 3
    n_frames = 60
    d_model = 128
    num_classes = 29

    # Create the model
    model = KeypointTransformer(n_keypoints=n_keypoints, n_features=n_features, n_frames=n_frames, d_model=d_model, num_classes=num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(MODEL_PATH))

    # Set the model to evaluation mode
    model.eval()
    return model


def made_pred(keypoints, model):
    #Make predictions (forward pass)
    #print(keypoints.shape, "pre_234")
    with torch.no_grad():  # Disable gradient calculations for inference
        #print(keypoints.shape, "prediction")
        output = model(keypoints)

    #probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
    predicted_classes = torch.argmax(output, dim=1)  # Get the predicted class (index with the highest score)
    #print(predicted_classes)
    # Print or return the predictions for the batch
    #print(f"Predicted probabilities: {probabilities}")
    #print(f"Predicted classes: {predicted_classes}")
    return predicted_classes


MODEL_PATH = 'keypoint_transformer.pth'



# Example keypoints input (similar shape to your training data)
# Replace this with actual input data
#keypoints = torch.randn(1, 60, 75, 3)
"""
# Replace 'your_file.json' with the path to your JSON file
with open('keypoints_dataset_test.json', 'r') as f:
    json_data = json.load(f)
# Move the model and input to the correct device (CPU or GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

for i in range(len(json_data)):
    d = json_data[i]['keypoints']

    #print(json_data[0]['keypoints'])
    tensor_1d = torch.tensor(d)
    if tensor_1d.shape[0] >1:
        tensor_split = torch.split(tensor_1d, 1, dim=0)
        for t in tensor_split:
            keypoints = tensor_1d.to(device)
            made_pred(keypoints)
    else:
        keypoints = tensor_1d.to(device)
        made_pred(keypoints)
"""





