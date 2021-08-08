

import shutil
import os
from pathlib import Path

source_dir = "./dataset/images_dataset"
target_train_dir = "./dataset/images_dataset_train_test/train"
target_test_dir = "./dataset/images_dataset_train_test/test"

Path(target_test_dir+"/A").mkdir(parents=True, exist_ok=True)
Path(target_train_dir+"/A").mkdir(parents=True, exist_ok=True)

directory_names = os.listdir(source_dir)
file_names = os.listdir(source_dir)


for index, dir_name in enumerate(directory_names):

    percent30 = 0
    percent70 = 0

    file_names = os.listdir(source_dir + "/" + dir_name)
    Path(target_test_dir + "/" + dir_name).mkdir(parents=True, exist_ok=True)
    Path(target_train_dir + "/" + dir_name).mkdir(parents=True, exist_ok=True)
    
    for index2, file_name in enumerate(file_names):
        if index2 < len(file_names) * 0.3:
            shutil.copy(os.path.join(source_dir + "/" + dir_name, file_name), target_test_dir + "/" + dir_name)
            percent30 += 1 
        else: 
            shutil.copy(os.path.join(source_dir + "/" + dir_name, file_name), target_train_dir + "/" + dir_name)
            percent70 += 1

    print("Letter: %s, percent70: %s, percent30: %s, len(file_names): %s"%(dir_name, percent70 , percent30, len(file_names)))

