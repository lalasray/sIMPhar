import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

simp = []
directory = r"C:\Users\lalas\Desktop\sIMphar\S10_2\mesh"
output_directory = r"C:\Users\lalas\Desktop\sIMphar"
directory_name = os.path.basename(os.path.dirname(directory))

files = os.listdir(directory)
files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
#print(files)

for filename in files:
    if filename.endswith(".txt"):
        frame_number = int(filename.split('_')[0])
        #print(frame_number)
        with open(os.path.join(directory, filename), 'r') as file:
            content = file.read()
            simp.append(content)
            #print(content,simp)

content_np =  np.array(simp)
file_name = directory_name + ".npy"
file_path = os.path.join(output_directory, file_name)
np.save(file_path, content_np)