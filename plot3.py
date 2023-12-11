import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('evaluation_results_2.csv')

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(df['Num cards'], df['Normalized score'], marker='o')

# Add title and labels
plt.title('Normalized Score vs. Number of Cards')
plt.xlabel('Num cards')
plt.ylabel('Normalized score')

# Show the plot
plt.savefig('plot.png')
plt.show()