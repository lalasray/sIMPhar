import pandas as pd
import matplotlib.pyplot as plt

# Create the DataFrame with the updated labels
data = {
    #"Percentage": ["20%", "40%", "60%", "80%", "100%"],
    #"No Pretrain": [0.218, 0.343, 0.452, 0.531, 0.581],
    #"Frozen Pretrain": [0.311, 0.425, 0.531, 0.582, 0.612],
    #"Late Learning": [0.335, 0.484, 0.644, 0.731, 0.767],
    "Embedding dim": [0.201, 0.489, 0.622, 0.724, 0.767, 0.779, 0.779],
    "dim": ["16", "32", "64", "128", "256", "512","1024"],
}

df = pd.DataFrame(data)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['dim'], df['Embedding dim'], marker='o')
#plt.plot(df['Percentage'], df['No Pretrain'], marker='o', label='No Pretrain')
#plt.plot(df['Percentage'], df['Frozen Pretrain'], marker='o', label='Frozen Pretrain')
#plt.plot(df['Percentage'], df['Late Learning'], marker='o', label='Late Learning')

plt.title('Training Set Performance')
plt.xlabel('Percentage')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.show()
