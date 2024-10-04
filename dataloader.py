from torch.utils.data import Dataset, DataLoader
import torch

def load_data(data):
    """
    Loads keypoints data into a DataLoader for training or inference.

    Args:
        data (list of dict): A list of dictionaries, where each dictionary contains:
            - 'keypoints': A list or nested list of keypoints for each frame.
            - 'label': The label (class) for the corresponding keypoints.

    Returns:
        DataLoader: A DataLoader object that provides batches of (keypoints, label) tuples.
    """
    
    # Step 1: Create a custom dataset class
    class KeypointDataset(Dataset):
        def __init__(self, data):
            """
            Initializes the dataset with the given data.

            Args:
                data (list of dict): List of data points where each point has 'keypoints' and 'label'.
            """
            self.data = data

        def __len__(self):
            """
            Returns the number of samples in the dataset.
            """
            return len(self.data)

        def __getitem__(self, idx):
            """
            Returns a single sample from the dataset at the specified index.

            Args:
                idx (int): Index of the sample.

            Returns:
                tuple: (keypoints, label) where:
                    - keypoints (torch.Tensor): Tensor of shape (n_frames, n_keypoints, n_features).
                    - label (torch.Tensor): The label for the sample as a long tensor.
            """
            sample = self.data[idx]
            
            # Convert keypoints and label to tensors
            keypoints = torch.tensor(sample['keypoints'], dtype=torch.float32)  # Shape: (n_frames, n_keypoints, n_features)
            label = torch.tensor(sample['label'], dtype=torch.long)  # Scalar representing the class label
            
            # Ensure keypoints are processed correctly if multiple tensors are present
            if keypoints.shape[0] > 1:
                tensor_split = torch.split(keypoints, 1, dim=0)  # Split along frames if needed
                keypoints = tensor_split[0]  # Select only the first chunk, if applicable
            
            return keypoints, label

    # Step 2: Create a dataset object
    keypoint_dataset = KeypointDataset(data)

    # Step 3: Create a DataLoader
    dataloader = DataLoader(keypoint_dataset, batch_size=12, shuffle=True)
    
    return dataloader
