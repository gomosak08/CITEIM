# Import necessary libraries
import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # MediaPipe for pose and hand keypoint extraction

# Initialize MediaPipe Holistic model and drawing utilities
mp_holistic = mp.solutions.holistic  # This provides holistic landmark detection (pose, hands, etc.)
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing the landmarks on the images

# Constants for keypoints
POSE_KEYPOINTS = 33  # MediaPipe detects 33 pose keypoints
HAND_KEYPOINTS = 21  # MediaPipe detects 21 keypoints for each hand
TOTAL_KEYPOINTS = POSE_KEYPOINTS + HAND_KEYPOINTS * 2  # Total keypoints (pose + both hands)
MAX_FRAMES = 60  # Number of frames to pad/truncate to

# Function to pad or split sequences of frames
def pad_or_split_keypoints(keypoints):
    """
    Pads or splits the keypoints data for each frame to ensure a consistent number of frames.
    If the sequence is shorter than MAX_FRAMES, it pads with zeros.
    If the sequence is longer, it splits the sequence into chunks of MAX_FRAMES.
    
    Args:
        keypoints (list): A list of keypoints, where each entry corresponds to a frame's keypoints.

    Returns:
        list: A list of chunks where each chunk is a list of keypoints for MAX_FRAMES frames.
    """
    num_frames = len(keypoints)
    chunks = []

    # If the video has fewer than MAX_FRAMES, pad with zero frames
    if num_frames < MAX_FRAMES:
        padding = [[[0.0, 0.0, 0.0]] * TOTAL_KEYPOINTS] * (MAX_FRAMES - num_frames)  # Zero padding
        keypoints += padding
        chunks.append(keypoints)  # Add the padded sequence as one chunk

    # If the video has more than MAX_FRAMES, split into chunks of 60 frames
    else:
        for i in range(0, num_frames, MAX_FRAMES):
            chunk = keypoints[i:i + MAX_FRAMES]  # Slice the keypoints into chunks

            # If the last chunk has fewer than 60 frames, pad it with zero frames
            if len(chunk) < MAX_FRAMES:
                padding = [[[0.0, 0.0, 0.0]] * TOTAL_KEYPOINTS] * (MAX_FRAMES - len(chunk))
                chunk += padding

            chunks.append(chunk)

    return chunks  # Return the list of chunks, where each chunk is MAX_FRAMES long


# Function to extract keypoints from the video
def extract_keypoints_from_video(path):
    """
    Extracts pose and hand keypoints from each frame of the video using MediaPipe's holistic model.
    
    Args:
        path (str): Path to the video file.
    
    Returns:
        list: A list of lists, where each inner list contains keypoints for each frame.
              The output is padded or split into chunks of MAX_FRAMES using the pad_or_split_keypoints() function.
    """
    video_path = path
    cap = cv2.VideoCapture(video_path)  # Open the video file using OpenCV
    keypoints = []  # List to hold keypoints for all frames

    # Initialize MediaPipe Holistic model for detecting pose and hand landmarks
    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()  # Read the next frame
            if not ret:
                break  # Break the loop if no more frames are available

            # Convert the frame from BGR (OpenCV format) to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to detect keypoints (pose and hands)
            result = holistic.process(rgb_frame)
            frame_keypoints = []  # Store keypoints for this specific frame

            # Extract pose keypoints if available
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])  # Append each pose landmark (x, y, z coordinates)
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * POSE_KEYPOINTS)  # Pad with zeros if no pose landmarks found

            # Extract left hand keypoints if available
            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])  # Append left hand landmarks
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * HAND_KEYPOINTS)  # Pad with zeros if no left hand landmarks

            # Extract right hand keypoints if available
            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    frame_keypoints.append([lm.x, lm.y, lm.z])  # Append right hand landmarks
            else:
                frame_keypoints.extend([[0.0, 0.0, 0.0]] * HAND_KEYPOINTS)  # Pad with zeros if no right hand landmarks

            # Append the keypoints for this frame to the overall list
            if frame_keypoints:
                keypoints.append(frame_keypoints)

    cap.release()  # Release the video file handle

    # Return the keypoints data padded or split into chunks of MAX_FRAMES
    return pad_or_split_keypoints(keypoints)
