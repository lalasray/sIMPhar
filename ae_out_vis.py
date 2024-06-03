import os
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

pickle_file_path = r"C:\Users\lalas\Desktop\n\out\S2_1.pt"

loaded_data = torch.load(pickle_file_path)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, figsize=(6, 6))
i = 0
j = 9600
# Plotting first subplot with dotted line (red)
axs[0].plot(loaded_data["IMP_mag"][i:])
#axs[0].plot(loaded_data["IMP_mag"], linestyle='--', label='GT magnitude')
axs[0].set_title('Magnitude')
axs[0].grid(True)
axs[0].legend()  # Adding legend to distinguish lines

axs[1].plot(loaded_data["sIMP_mag"][i:])
#axs[0].plot(loaded_data["IMP_mag"], linestyle='--', label='GT magnitude')
axs[1].grid(True)

# Plotting second subplot with dotted line (red)
axs[2].plot(loaded_data["IMP_phase"][i:])
#axs[1].plot(loaded_data["IMP_mag"], label='GT phase', linestyle='--')
axs[2].set_title('Phase')
axs[2].grid(True)
axs[2].legend()  # Adding legend to distinguish lines

axs[3].plot(loaded_data["sIMP_phase"][i:])
#axs[1].plot(loaded_data["IMP_mag"], label='GT phase', linestyle='--')
axs[3].grid(True)


plt.tight_layout()
plt.show()
