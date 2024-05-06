import os
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

pickle_file_path = r"C:\Users\lalas\Desktop\sIMphar\out\S10_1.pt"

loaded_data = torch.load(pickle_file_path)

fig, axs = plt.subplots(5, 1, figsize=(20, 6))

axs[0].plot(loaded_data["sIMP_mag"])
axs[0].set_title('synth mag')
axs[0].grid(True)

axs[1].plot(loaded_data["IMP_mag"])
axs[1].set_title('real mag')
axs[1].grid(True)

axs[2].plot(loaded_data["sIMP_phase"])
axs[2].set_title('synth phase')
axs[2].grid(True)

axs[3].plot(loaded_data["IMP_phase"])
axs[3].set_title('real phase')
axs[3].grid(True)

axs[4].plot(loaded_data["class"])
axs[4].set_title('class')
axs[4].grid(True)


plt.tight_layout()
plt.show()
