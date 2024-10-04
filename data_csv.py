import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from extract_key_points import extract_keypoints_from_video
import warnings
warnings.filterwarnings("ignore")
import json


main_folder = "/home/gomosak/CITEIM/10_09_24_v1/"

dataset = []
class_folders = sorted(os.listdir(main_folder))  # Assumes folders are labeled with class names

for label, class_folder in enumerate(class_folders):
    class_folder_path = os.path.join(main_folder, class_folder)
    if os.path.isdir(class_folder_path):
        # Process each video in the class folder
        for video_file in os.listdir(class_folder_path)[:1]:
            video_path = os.path.join(class_folder_path, video_file)
            
            if video_path.endswith(".mp4"):  # Ensure it is a video file
                print(f"Processing video: {video_path}")

                # Extract keypoints from the video
                keypoints = extract_keypoints_from_video(video_path)
                #print(f'keypoints: {keypoints}, label: {label}')
                # Append keypoints and label to the dataset
                if keypoints:
                    dataset.append({"keypoints": keypoints, "label": label}) 


def save_dataset(dataset, output_file):
    with open(output_file, 'w') as f:
        json.dump(dataset, f)

output_file = "keypoints_dataset_test.json"
save_dataset(dataset, output_file)
