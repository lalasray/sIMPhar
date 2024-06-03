import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your actual CSV file
csv_file_path = r"C:\Users\lalas\Desktop\Rec1-2024-04-05_12-11-59\Light.csv"

# Read the CSV data into a DataFrame
df = pd.read_csv(csv_file_path)

# Filter the DataFrame for frames between 100 and 500 (inclusive)
df_filtered = df.iloc[:]  # iloc is used for integer-location based indexing
df_filtered['lux_smoothed'] = df_filtered['lux'].rolling(window=10).mean()
# Plotting the filtered data
plt.figure(figsize=(10, 6))
plt.plot(df_filtered.index, df_filtered['lux'].clip(0, 800))
plt.xlabel('Frame')
plt.ylabel('Lux')
plt.title('Lux over Time')
plt.grid(True)
plt.show()
