# Import necessary libraries and modules
from extract_key_points import extract_keypoints_from_video  # Function to extract keypoints from a video
import json  # For handling JSON data
from prediction import made_pred, load_model  # Custom functions: made_pred for making predictions, load_model for loading a model
import torch  # PyTorch library for tensor operations and model handling
import warnings  # Used to control the display of warnings

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

# Load class/label mappings from a JSON file
with open('data.json', 'r') as f:
    data = json.load(f)  # The JSON file contains a dictionary that maps prediction indices to human-readable labels

# Determine the device (GPU or CPU) to run the model on
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pre-trained model
model = load_model()  # Custom function to load the model (implementation not provided)
model = model.to(device)  # Move the model to the appropriate device (GPU/CPU)

# Get the path of the video to classify from user input
video_path = input("Give me the path to the video to classify: ")

# Check if the input file is a valid video file (assumed to be .mp4 format)
if video_path.endswith(".mp4"):  
    print(f"Processing video: {video_path}")

    # Extract keypoints from the video using the custom function
    keypoints = extract_keypoints_from_video(video_path)

    # Convert the keypoints into a PyTorch tensor for model input
    tensor_1d = torch.tensor(keypoints)

    # Ensure there is more than one keypoint in the tensor
    if tensor_1d.shape[0] > 1:
        # Split the tensor into smaller chunks (1 sample at a time)
        tensor_split = torch.split(tensor_1d, 1, dim=0)
        
        # Move the keypoints tensor to the appropriate device (GPU/CPU)
        keypoints = tensor_1d.to(device)
        
        # Pass the keypoints through the model to make a prediction
        pred = made_pred(keypoints, model)  # Custom function for prediction
        
        # Convert the model's prediction (PyTorch tensor) to a Python list
        integer_list = pred.tolist()

        # Output the prediction results as class labels
        print(integer_list)
        for i in integer_list:
            print(data[str(i)])  # Print the corresponding label from the loaded data.json
