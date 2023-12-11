import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('evaluation_results.csv')

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(df['Num episodes (train)'], df['Normalized score'], marker='o')

# Add title and labels
plt.title('Degree of Optimality vs. Training Episodes\nfor a 40-Flashcard Deck')
plt.xlabel('Num episodes (train)')
plt.ylabel('Degree of optimality')

# Show the plot
plt.savefig('plot.png')
plt.show()