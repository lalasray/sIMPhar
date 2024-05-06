import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import torch

def smooth_data(data, window_size=15):
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        start_index = max(0, i - window_size)
        end_index = min(len(data), i + window_size + 1)
        smoothed_data[i] = np.mean(data[start_index:end_index])
    return smoothed_data

def lowpass_filter(freq,fc,data):
    w = fc / (freq / 2) # Normalize the frequency
    b, a = butter(2, w, 'low')
           
    filtered_data = filtfilt(b, a, data)
   
    return filtered_data

sIMP = np.load(r"C:\Users\lalas\Desktop\sIMphar\S10_2.npy")
sIMP = np.array(sIMP).astype(float)
datapath = r"C:\Users\lalas\Downloads\EIS_p09_1_1713968563618.txt"
IMP = pd.read_csv(datapath, sep=",", usecols=[3, 4, 5], index_col=False)

    
fig, axs = plt.subplots(3, 1, figsize=(20, 6))
#zero_array = np.zeros(5500)
#sIMP = np.concatenate((zero_array, sIMP))
# Plot 1: synth 1
axs[0].plot(sIMP[0:])
axs[0].set_title('synth 1')
axs[0].grid(True)

# Plot 2: real mag
axs[1].plot(IMP.iloc[:, 0])
axs[1].set_title('real mag')
axs[1].set_ylim(IMP.iloc[:, 0].min(), IMP.iloc[:, 0].max())
axs[1].grid(True)

# Plot 3: real phase
axs[2].plot(IMP.iloc[:, 1])
axs[2].set_title('real phase')
axs[2].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

sIMP_new = sIMP[0:]
IMP_mag = np.array(IMP.iloc[:, 0])
IMP_phase = np.array(IMP.iloc[:, 1])
IMP_class = np.array(IMP.iloc[:,2])

interp_func = interp1d(np.linspace(0, 1, len(sIMP_new)), sIMP_new)
interpolated_sIMP_new = interp_func(np.linspace(0, 1, len(IMP_mag)))

fig, axs = plt.subplots(3, 1, figsize=(20, 6))

axs[0].plot(interpolated_sIMP_new)
axs[0].set_title('synth 1')
axs[0].grid(True)

axs[1].plot(IMP_mag)
axs[1].set_title('real mag')
#axs[1].set_ylim(IMP.iloc[:, 0].min(), IMP.iloc[:, 0].max())
axs[1].grid(True)

axs[2].plot(IMP_phase)
axs[2].set_title('real phase')
axs[2].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

sIMP_new_tensor = torch.tensor(sIMP_new)
IMP_mag_tensor = torch.tensor(IMP_mag)
IMP_phase_tensor = torch.tensor(IMP_phase)
IMP_class_tensor = torch.tensor(IMP_class)

# Create a dictionary to store tensors
pk_dict = {
    'sIMP': sIMP_new_tensor,
    'IMP_mag': IMP_mag_tensor,
    'IMP_phase': IMP_phase_tensor,
    'class': IMP_class_tensor
}

# Save dictionary containing tensors
torch.save(pk_dict, r'C:\Users\lalas\Desktop\sIMphar\S10_2.pt')