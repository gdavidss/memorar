import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('evaluation_result_2.csv')

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(df['Num episodes (train)'], df['Model utility'], marker='o', label='Model utility')
plt.plot(df['Num episodes (train)'], df['Optimal policy utility'], marker='o', label='Optimal policy utility')
plt.plot(df['Num episodes (train)'], df['Random Policy Utility'], marker='o', label='Random Policy Utility')

# Add title and labels
plt.title('Utility Comparison vs. Training Episodes\nfor a 40-Flashcard Deck')
plt.xlabel('Num episodes (train)')
plt.ylabel('Utility')
plt.legend()

# Show the plot
plt.savefig('plot_2.png')
plt.show()