import os
import cv2
import pandas as pd
import time

time_start = time.time()

root_folder = r'C:\Users\szawe\Desktop\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'
output_folder = r'C:\Users\szawe\Desktop\Output'
size = (64, 64)
format = '.png'
interpolation = cv2.INTER_CUBIC
# interpolation = cv2.INTER_LINEAR_EXACT
# interpolation = cv2.INTER_LINEAR

for subdir, dirs, files in os.walk(root_folder):
    csv_file = None
    for file in files:
        if file.endswith('.csv'):
            csv_file = os.path.join(subdir, file)
            break
    if not csv_file:
        continue

    df = pd.read_csv(csv_file, sep=';')

    relative_subdir = os.path.relpath(subdir, root_folder)
    output_subfolder = os.path.join(output_folder, relative_subdir)
    os.makedirs(output_subfolder, exist_ok=True)

    for index, row in df.iterrows():
        filename = row['Filename']
        image = cv2.imread(os.path.join(subdir, filename))

        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, size, interpolation=interpolation)

        path = os.path.join(output_subfolder, f"cropped_{filename.split('.')[0]}{format}")
        cv2.imwrite(path, resized)

    print(f"Processed {subdir}")

print(f'Time: {time.time() - time_start}')