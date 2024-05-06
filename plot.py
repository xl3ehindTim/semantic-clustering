import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Needed for annotations

# Sample data (replace with your actual word embeddings)
words = ["apple", "banana", "orange", "chair", "table"]
embeddings = [[0.2, 0.5], [-0.1, 0.7], [0.4, 0.3], [-0.5, -0.2], [-0.3, 0.1]]

# Create a DataFrame
data = pd.DataFrame(list(zip(words, embeddings)), columns = ['word', 'x', 'y'])

# Create the scatter plot
sns.scatterplot(
    x = "x",
    y = "y",
    hue = "word",  # Color points by word
    data = data,
    style = "word"  # Add markers based on word
)

# Option 1: Using annotations (more control)
# for i, row in data.iterrows():
#   plt.annotate(row['word'], (row['x'], row['y']), fontsize=8)  # Adjust fontsize as needed

# Option 2: Using ax.text (simpler, less control)
ax = sns.plt.gca()
for i, row in data.iterrows():
  ax.text(row['x'], row['y'], row['word'], ha='center', va='center', fontsize=8)  # Adjust fontsize as needed

# Customize the plot (optional)
sns.title("Sample Word Embeddings")
sns.xlabel("Dimension 1")
sns.ylabel("Dimension 2")
plt.show()
