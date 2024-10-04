from extract_key_points import extract_keypoints_from_video
import json
from prediction import made_pred, load_model
import torch
import warnings
warnings.filterwarnings("ignore")



with open('data.json', 'r') as f:
    data = json.load(f)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model()
model = model.to(device)

video_path = input("Give me the path to the video to classify: ")

if video_path.endswith(".mp4"):  # Ensure it is a video file
    print(f"Processing video: {video_path}")

    # Extract keypoints from the video
    keypoints = extract_keypoints_from_video(video_path)

    tensor_1d = torch.tensor(keypoints)
    if tensor_1d.shape[0] >1:
        tensor_split = torch.split(tensor_1d, 1, dim=0)
        keypoints = tensor_1d.to(device)
        pred = made_pred(keypoints, model)
        integer_list = pred.tolist()

        print(integer_list)
        for i in integer_list:
            print(data[str(i)])

