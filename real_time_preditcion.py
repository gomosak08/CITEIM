import cv2
import mediapipe as mp
import time
from prediction import made_pred, load_model
import torch
import json
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")
# Constants for keypoints and frames
POSE_KEYPOINTS = 33
HAND_KEYPOINTS = 21
TOTAL_KEYPOINTS = POSE_KEYPOINTS + HAND_KEYPOINTS * 2
MAX_FRAMES = 60  # Fixed number of frames

with open('data.json', 'r') as f:
    data = json.load(f)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Drawing Utilities for visualization
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
keypoints = []


def prediction(keypoints, model):
    tensor_1d = torch.tensor(keypoints)
    keypoints = tensor_1d.to(device)
    print(keypoints.shape)
    pred = made_pred(keypoints, model)
    #integer_list = pred.tolist()
    #for i in integer_list:
    #    print(data[str(i)])

def main(model):
    # Create a holistic object for pose, hands, and face landmarks
    with mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        frame_num = 0
        empempty_list = [[[0.0, 0.0, 0.0] for _ in range(75)] for _ in range(60)]
        segmento = [empempty_list,empempty_list,empempty_list,empempty_list]


        start_time = time.time()  # Record the start time
        last_check_time = start_time   # Initialize the last check time

        data_storage = [] 
        keypoints = []

        n_list = 0
        while cap.isOpened():
            current_time = time.time()  # Get the current time
            # Check if 0.25 seconds have passed since the last check
            if current_time - last_check_time >= 0.25:
                print(current_time - last_check_time > 0.25, current_time, last_check_time, last_check_time+5)
                segmento[n_list] = data_storage[-60:]
                n_list += 1
                n_list = n_list%4
                
                print(len(segmento[n_list]))
                prediction(segmento[n_list],model)

                #print(current_time - last_check_time, n_list)
                #elapsed_time = current_time - start_time
                #print(f"Elapsed time: {elapsed_time:.2f} seconds")
                last_check_time = current_time

            if frame_num%120 == 0:
                data_storage.extend(keypoints)
                keypoints = []
                L = len(data_storage)
                if L >90:
                    to_del = L-90
                    del data_storage[:to_del]
                print(f'frame {frame_num}')
                print(len(data_storage))          

            frame_num += 1
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and find all the landmarks
            result = holistic.process(rgb_frame)
            frame_keypoints = []

            # Extract pose keypoints
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * POSE_KEYPOINTS)

            # Extract left hand keypoints
            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])
                mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * HAND_KEYPOINTS)

            # Extract right hand keypoints
            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])
                mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * HAND_KEYPOINTS)

            # Optionally extract face keypoints and add zero padding similarly if necessary

            # Append keypoints for the current frame
            if frame_keypoints:
                keypoints.append(frame_keypoints)
                #print(frame_num)

            #gray = cv2.cvtColor(frame)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(len(segmento[0]),len(data_storage))

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model()
    model = model.to(device)
    main(model)

    #start_time = time.time()  # Record the start time
    #last_check_time = start_time 
    #time.sleep(.30)
    #get_list(last_check_time)