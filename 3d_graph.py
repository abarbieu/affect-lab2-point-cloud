import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display



n_points = 1000  # Define the total number of points to be generated
start_time = datetime.now()  # Define the start time
timestamps = [start_time + timedelta(milliseconds=x) for x in range(n_points)]

P = np.random.uniform(-1, 1, n_points)
A = np.random.uniform(-1, 1, n_points)
D = np.random.uniform(-1, 1, n_points)

df = pd.DataFrame({
    'Timestamp': timestamps,
    'P': P,
    'A': A,
    'D': D
})


# Filter the DataFrame to only include the first two seconds of data
two_seconds = timedelta(seconds=2)
df_2_sec = df[df['Timestamp'] < start_time + two_seconds]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add the data to the plot
ax.scatter(df_2_sec['P'], df_2_sec['A'], df_2_sec['D'])

# Add labels to the axes
ax.set_xlabel('P')
ax.set_ylabel('A')
ax.set_zlabel('D')

# Display the plot
plt.show()

# Calculate the milliseconds since the start time for each timestamp
df['Milliseconds'] = (df['Timestamp'] - start_time).dt.total_seconds() * 1000

# Calculate the 2-second chunk index
df['Chunk_Index'] = (df['Milliseconds'] // 2000).astype(int)


