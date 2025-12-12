import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

path = 'Images'  # Folder containing the images
img_dimensions = (64, 64) # Force all images to be small squares


if not os.path.exists(path):
    print(f"Error: Could not find folder '{path}'. Did you extract the zip here?")
    exit()

myList = os.listdir(path) # Count Classes
noOfClasses = len(myList)
print(f"Detected {noOfClasses} different types of traffic signs.")

# Load the Images
images = []
classNo = []

# Get the actual list of folder names
folder_names = os.listdir(path)

# Loop through the FOLDERS
for folder in folder_names:
    if not os.path.isdir(path + "/" + folder):
        continue
        
    # Get the Class ID from the folder name
    try:
        current_class = int(folder) 
    except:
        continue # Skip folders that aren't numbers
        
    myPicList = os.listdir(path + "/" + folder)
    
    # Load images from this folder
    for y in myPicList:
        try:
            curImg = cv2.imread(path + "/" + folder + "/" + y)
            curImg = cv2.resize(curImg, img_dimensions)
            images.append(curImg)
            classNo.append(current_class)
        except:
            print(f"Error reading image: {y}")
            
    print(f"Class {current_class} Loaded")

print(f"\nData Loading Complete. Found {len(images)} images.")
images = np.array(images)
classNo = np.array(classNo)

import random

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    # Pick a random image from our loaded list
    index = random.randint(0, len(images)-1)
    
    # Matplotlib needs RGB, OpenCV gives BGR. We convert it.
    axes[i].imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"Class ID: {classNo[index]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()