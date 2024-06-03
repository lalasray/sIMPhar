import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your actual CSV file
csv_file_path = r"C:\Users\lalas\Desktop\Rec1-2024-04-05_12-11-59\Accelerometer.csv"

# Read the CSV data into a DataFrame
df = pd.read_csv(csv_file_path)

# Filter the DataFrame for frames between 100 and 500 (inclusive)
df_filtered = df.iloc[1500:1600]  # iloc is used for integer-location based indexing

# Apply smoothing using a rolling mean with a window size of 5
df_filtered['z_smoothed'] = df_filtered['z'].rolling(window=5).mean()
df_filtered['y_smoothed'] = df_filtered['y'].rolling(window=5).mean()
df_filtered['x_smoothed'] = df_filtered['x'].rolling(window=5).mean()

# Plotting the filtered and smoothed data
plt.figure(figsize=(10, 6))

plt.plot(df_filtered.index, df_filtered['z_smoothed'].clip(-10, 10), label='z')
plt.plot(df_filtered.index, df_filtered['y_smoothed'].clip(-10, 10), label='y')
plt.plot(df_filtered.index, df_filtered['x_smoothed'].clip(-10, 10), label='x')

plt.xlabel('Frame')
plt.ylabel('Values')
plt.title('Accelometer Data over Time')
plt.legend()
plt.grid(True)
plt.show()
