import os
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

pickle_file_path = r"C:\Users\lalas\Desktop\n\out\S1_1.pt"

loaded_data = torch.load(pickle_file_path)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(24, 12))

# Plotting first subplot with dotted line (red)
axs[0].plot(loaded_data["sIMP_mag"], label='sIMP magnitude', color='black')
axs[0].plot(loaded_data["IMP_mag"], linestyle='--', label='GT magnitude')
axs[0].set_title('Magnitude')
axs[0].grid(True)
axs[0].legend()  # Adding legend to distinguish lines

# Plotting second subplot with dotted line (red)
axs[1].plot(loaded_data["sIMP_phase"], color='black', label='sIMP phase')
axs[1].plot(loaded_data["IMP_mag"], label='GT phase', linestyle='--')
axs[1].set_title('Phase')
axs[1].grid(True)
axs[1].legend()  # Adding legend to distinguish lines

plt.tight_layout()
plt.show()
