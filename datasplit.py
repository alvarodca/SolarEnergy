import os
from glob import glob
import random
import shutil


# Folders
original_folder = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\curve_images"

# Training
training_folder_0 = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\data\training\0"
training_folder_1 = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\data\training\1"
training_folder_2 = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\data\training\2"

# Validation
validation_folder_0 = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\data\validation\0"
validation_folder_1 = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\data\validation\1"
validation_folder_2 = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\data\validation\2"

# Testing
testing_folder = r"C:\Users\1\Documents\SolarEnergy\SolarEnergy\data\testing"



# Making sure the directories appear
os.makedirs(training_folder_0, exist_ok=True)
os.makedirs(training_folder_1, exist_ok=True)
os.makedirs(training_folder_2, exist_ok=True)
os.makedirs(validation_folder_0, exist_ok=True)
os.makedirs(validation_folder_1, exist_ok=True)
os.makedirs(validation_folder_2, exist_ok=True)
os.makedirs(testing_folder,exist_ok=True)



# Obtaining images
images = glob(os.path.join(original_folder,"*.png"))
# Shuffling data
random.shuffle(images)

# 80/10/10 split ratio
split_1 = int(0.8 * len(images))
split_2 = int(0.9 * len(images))

# Subsets
training = images[:split_1]
validation = images[split_1:split_2]
testing = images[split_2:]

# Obtaining training, validation and testing data, they each have a separate folder per label to simplify DataLoader processes later
for file in training:
    if "Label_0" in file:
        shutil.copy(file,training_folder_0)

    elif "Label_1" in file:
        shutil.copy(file, training_folder_1)

    else:
        shutil.copy(file, training_folder_2)

for file in validation:
    if "Label_0" in file:
        shutil.copy(file, validation_folder_0)

    elif "Label_1" in file:
        shutil.copy(file, validation_folder_1)

    else:
        shutil.copy(file, validation_folder_2)

for file in testing:
    shutil.copy(file, testing_folder)
