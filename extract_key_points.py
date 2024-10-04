import cv2
import mediapipe as mp 

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Constants
POSE_KEYPOINTS = 33  # 33 keypoints for pose
HAND_KEYPOINTS = 21  # 21 keypoints for each hand (left and right)
TOTAL_KEYPOINTS = POSE_KEYPOINTS + HAND_KEYPOINTS * 2  # Total keypoints for pose + hands
MAX_FRAMES = 60  # Number of frames to pad/truncate to

# Function to pad or split sequences of frames
def pad_or_split_keypoints(keypoints):
    num_frames = len(keypoints)
    chunks = []

    # If the video has fewer than MAX_FRAMES, pad with zero frames
    if num_frames < MAX_FRAMES:
        padding = [[[0.0, 0.0, 0.0]] * TOTAL_KEYPOINTS] * (MAX_FRAMES - num_frames)
        keypoints += padding
        chunks.append(keypoints)  # Add the padded sequence as one chunk

    # If the video has more than MAX_FRAMES, split into chunks of 60 frames
    else:
        for i in range(0, num_frames, MAX_FRAMES):
            chunk = keypoints[i:i + MAX_FRAMES]

            # If the last chunk has fewer than 60 frames, pad it with zero frames
            if len(chunk) < MAX_FRAMES:
                padding = [[[0.0, 0.0, 0.0]] * TOTAL_KEYPOINTS] * (MAX_FRAMES - len(chunk))
                chunk += padding

            chunks.append(chunk)

    return chunks

# Function to extract keypoints from the video
def extract_keypoints_from_video(path):
    video_path = path
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    # Create a holistic object for pose, hands, and face landmarks
    with mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
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
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * POSE_KEYPOINTS)

            # Extract left hand keypoints
            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * HAND_KEYPOINTS)

            # Extract right hand keypoints
            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * HAND_KEYPOINTS)

            # Optionally extract face keypoints and add zero padding similarly if necessary

            # Append keypoints for the current frame
            if frame_keypoints:
                keypoints.append(frame_keypoints)

    cap.release()

    # Pad or split the keypoints into chunks of MAX_FRAMES
    return pad_or_split_keypoints(keypoints)