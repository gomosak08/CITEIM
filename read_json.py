import json
import torch

# Replace 'your_file.json' with the path to your JSON file
with open('keypoints_dataset.json', 'r') as f:
    json_data = json.load(f)

# Print the first dictionary in the list
"""if isinstance(json_data, list) and len(json_data) > 0:
    print(json_data[-1])
else:
    print("The JSON file is either empty or not a list.")"""




print(len(json_data))
print(json_data[-1])
"""for i in range(15):
    d = json_data[i]['keypoints']

    #print(json_data[0]['keypoints'])
    tensor_1d = torch.tensor(d)
    if tensor_1d.shape[0] >1:
        tensor_split = torch.split(tensor_1d, 1, dim=0)
        for t in tensor_split:
            print(t.shape)
    else:
        print(tensor_1d.shape)
"""
#75