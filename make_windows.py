import os
import numpy as np
import torch
from InstructorEmbedding import INSTRUCTOR
data = "simp" #simp #imp

def majority_vote(tensor):
    
    unique_values, counts = tensor.unique(return_counts=True)
    #print(unique_values, counts)
    majority_index = counts.argmax()
    majority_class = unique_values[majority_index]
    return majority_class.type(torch.int8)

if data == "simp":
    
    imp_files = []
    pose_window_size = 50 
    pose_stride = 10
    
    path = r"C:\Users\lalas\Desktop\n\out"
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pt'):
                imp_files.append(os.path.join(root, file))
    
    #print(imp_files)
        

    for idx in range(len(imp_files)):
        tensor = torch.load(imp_files[idx])
        #print(tensor)
        simp_mag = tensor["sIMP_mag"]
        simp_phase = tensor["sIMP_phase"]
        classes = tensor["class"]
        #imp_mag = tensor["imp_mag"]
        #imp_phase = tensor["imp_phase"]
        imp_mag_windows = [simp_mag[i:i+pose_window_size] for i in range(0, len(simp_mag) - pose_window_size + 1, pose_stride)]
        imp_phase_windows = [simp_phase[i:i+pose_window_size] for i in range(0, len(simp_phase) - pose_window_size + 1, pose_stride)]
        class_windows = [classes[i:i+pose_window_size] for i in range(0, len(classes) - pose_window_size + 1, pose_stride)]

        for i in range (len(imp_mag_windows)):
            x = majority_vote(class_windows[i])
            if x == 0:
                sentence = [['Represent human activity sentence for clustering: ','Doing random activities.']]
            elif x == 1:
                sentence = [['Represent human activity sentence for clustering: ','Boxing with both hands.']]
            elif x == 2:
                sentence = [['Represent human activity sentence for clustering: ','Doing Biceps Curls.']]
            elif x == 3:
                sentence = [['Represent human activity sentence for clustering: ',"Doing Chest Press."]]
            elif x == 4:
                sentence = [['Represent human activity sentence for clustering: ','Doing Shoulder and Chest Press.']]
            elif x == 5:
                sentence = [['Represent human activity sentence for clustering: ','Doing Arm hold and Shoulder Press']]
            elif x == 6:
                sentence = [['Represent human activity sentence for clustering: ','Arm Opener.']]
            elif x == 7:
                sentence = [['Represent human activity sentence for clustering: ','Answering telephone.']]
            elif x == 8:
                sentence = [['Represent human activity sentence for clustering: ','Wearing VR headsets.']]
            elif x == 9:
                sentence = [['Represent human activity sentence for clustering: ','Sweeping table.']]
            

            model = INSTRUCTOR('hkunlp/instructor-large')
            embeddings = model.encode(sentence)
            tensor_dict = {
            "simp_mag": imp_mag_windows[i],
            "simp_phase": imp_phase_windows[i],
            "embedding": embeddings}
            n_root = os.path.dirname(imp_files[idx])
            save_path = n_root+'/synth/'
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.splitext(os.path.basename(imp_files[idx]))[0]
            torch.save(tensor_dict, os.path.join(save_path, f"{filename}_window{i}.pth"))
            print(save_path)
            

elif data == "imp":
    
    imp_files = []
    pose_window_size = 50 
    pose_stride = 10
    
    path = r"C:\Users\lalas\Desktop\n\out"
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pt'):
                imp_files.append(os.path.join(root, file))
    
    #print(imp_files)
        

    for idx in range(len(imp_files)):
        tensor = torch.load(imp_files[idx])
        #print(tensor)
        simp_mag = tensor["IMP_mag"]
        simp_phase = tensor["IMP_phase"]
        classes = tensor["class"]
        #imp_mag = tensor["imp_mag"]
        #imp_phase = tensor["imp_phase"]
        imp_mag_windows = [simp_mag[i:i+pose_window_size] for i in range(0, len(simp_mag) - pose_window_size + 1, pose_stride)]
        imp_phase_windows = [simp_phase[i:i+pose_window_size] for i in range(0, len(simp_phase) - pose_window_size + 1, pose_stride)]
        class_windows = [classes[i:i+pose_window_size] for i in range(0, len(classes) - pose_window_size + 1, pose_stride)]

        #print(len(imp_mag_windows), len(imp_mag_windows),len(class_windows))
        for i in range (len(imp_mag_windows)):
            x = majority_vote(class_windows[i])
            if x == 0:
                sentence = [['Represent human activity sentence for clustering: ','Doing random activities.']]
            elif x == 1:
                sentence = [['Represent human activity sentence for clustering: ','Boxing with both hands.']]
            elif x == 2:
                sentence = [['Represent human activity sentence for clustering: ','Doing Biceps Curls.']]
            elif x == 3:
                sentence = [['Represent human activity sentence for clustering: ',"Doing Chest Press."]]
            elif x == 4:
                sentence = [['Represent human activity sentence for clustering: ','Doing Shoulder and Chest Press.']]
            elif x == 5:
                sentence = [['Represent human activity sentence for clustering: ','Doing Arm hold and Shoulder Press']]
            elif x == 6:
                sentence = [['Represent human activity sentence for clustering: ','Arm Opener.']]
            elif x == 7:
                sentence = [['Represent human activity sentence for clustering: ','Answering telephone.']]
            elif x == 8:
                sentence = [['Represent human activity sentence for clustering: ','Wearing VR headsets.']]
            elif x == 9:
                sentence = [['Represent human activity sentence for clustering: ','Sweeping table.']]
            

            model = INSTRUCTOR('hkunlp/instructor-large')
            embeddings = model.encode(sentence)
            tensor_dict = {
            "simp_mag": imp_mag_windows[i],
            "simp_phase": imp_phase_windows[i],
            "aclass": x,
            "embedding": embeddings}
            n_root = os.path.dirname(imp_files[idx])
            save_path = n_root+'/real/'
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.splitext(os.path.basename(imp_files[idx]))[0]
            torch.save(tensor_dict, os.path.join(save_path, f"{filename}_window{i}.pth"))
            print(save_path)