import os
import pandas as pd
import cv2
import time

root_folder = ".\\images\\input"
output_folder = '.\\images\\output'
size = (256, 256)  # specify the desired size
interpolation = cv2.INTER_LINEAR  # specify the interpolation method
format = '.jpg'  # specify the desired output format

time_start = time.time()

for subdir, _, files in os.walk(root_folder):

    for file in files:
        file_path = os.path.join(subdir, file)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                resized_img = cv2.resize(img, size, interpolation=interpolation)
                output_file = os.path.splitext(file)[0] + format
                output_path = os.path.join(output_folder, output_file)
                
                cv2.imwrite(output_path, resized_img)
                


    print(f"Processed {subdir}")

print(f'Time: {time.time() - time_start}')