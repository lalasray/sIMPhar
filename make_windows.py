import os
import numpy as np
import torch
from InstructorEmbedding import INSTRUCTOR
data = "simp" #simp #imp

if data == "simp":
    imp_files = []
    pose_window_size = 50 
    pose_stride = 10
    path = r"/media/lala/Seagate/Dataset/Meta/sIMphar_synth/sIMphar_synth" 

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pt'):
                imp_files.append(os.path.join(root, file))

    #print(imp_files)

    for idx in range(len(imp_files)):
        tensor = torch.load(imp_files[idx])
        #print(tensor.shape)

        imp_windows = [tensor[i:i+pose_window_size] for i in range(0, len(tensor) - pose_window_size + 1, pose_stride)]
        i = 0
        #print(len(imp_windows))
        for window in imp_windows:
            i = i+1
            #print(window)
            imp = torch.tensor(window)
            model = INSTRUCTOR('hkunlp/instructor-large')
            n_root = os.path.dirname(imp_files[idx])
            with open(n_root+"/results.txt", 'r') as file:
                # Read the first line
                first_line = file.readline().strip()  # Strip() removes any leading/trailing whitespace
            sentence = ['Represent human activity sentence for clustering: ',first_line]
            embeddings = model.encode(sentence)
            text = embeddings
            # Save tensors
            save_path = n_root+'tensors/'
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.splitext(os.path.basename(imp_files[idx]))[0]
            torch.save((imp,text), os.path.join(save_path, f"{filename}_window{i}.pt"))
            print(save_path)
