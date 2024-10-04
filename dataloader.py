from torch.utils.data import Dataset, DataLoader
import torch


def load_data(data):
    # Step 1: Create a custom dataset class
    class KeypointDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            keypoints = torch.tensor(sample['keypoints'], dtype=torch.float32)  # Convert to tensor
            label = torch.tensor(sample['label'], dtype=torch.long)  # Convert label to tensor
            #print(keypoints.shape)
            if keypoints.shape[0] >1:
                tensor_split = torch.split(keypoints, 1, dim=0)
                keypoints = tensor_split[0]
            return keypoints, label

    # Step 2: Create a dataset object
    keypoint_dataset = KeypointDataset(data)

    # Step 3: Create a DataLoader
    dataloader = DataLoader(keypoint_dataset, batch_size=12, shuffle=True)
    return dataloader

