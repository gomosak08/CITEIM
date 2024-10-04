import os
import json


main_folder = "/home/gomosak/CITEIM/10_09_24_v1/"
class_folders = sorted(os.listdir(main_folder))

data = {}
for label, class_folder in enumerate(class_folders):
    print(label,class_folder)
    data[label] = class_folder


with open('data.json', 'w') as f:
    json.dump(data, f)