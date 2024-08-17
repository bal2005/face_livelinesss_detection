import random
import os
import shutil
from itertools import islice

outputFolderPath = "D:\\dataset\\datacollect\\splitdata3"
inputFolderPath = "D:\\dataset\\datacollect"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# Remove existing output folder and create new directories
try:
    shutil.rmtree(outputFolderPath)
except OSError:
    pass  # Directory does not exist, no need to create it

os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# Get the names of files (excluding directories)
listNames = [name for name in os.listdir(
    inputFolderPath) if os.path.isfile(os.path.join(inputFolderPath, name))]

# Extract unique names without extensions
uniqueNames = list(set(name.split('.')[0]
                   for name in listNames if name.endswith('.jpg')))

# Shuffle and split data
random.shuffle(uniqueNames)
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

if lenData != lenTrain + lenTest + lenVal:
    lenTrain += lenData - (lenTrain + lenTest + lenVal)

lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \nSplit: {
      len(Output[0])} {len(Output[1])} {len(Output[2])}')

# Copy files to respective directories
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        src_img = os.path.join(inputFolderPath, f'{fileName}.jpg')
        dst_img = os.path.join(
            outputFolderPath, sequence[i], 'images', f'{fileName}.jpg')
        if os.path.isfile(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f'Image file not found: {src_img}')

        src_txt = os.path.join(inputFolderPath, f'{fileName}.txt')
        dst_txt = os.path.join(
            outputFolderPath, sequence[i], 'labels', f'{fileName}.txt')
        if os.path.isfile(src_txt):
            shutil.copy(src_txt, dst_txt)
        else:
            print(f'Label file not found: {src_txt}')

print("Split Process Completed...")

# Create data.yaml
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

with open(os.path.join(outputFolderPath, 'data.yaml'), 'w') as f:
    f.write(dataYaml)

print("Data.yaml file Created...")
