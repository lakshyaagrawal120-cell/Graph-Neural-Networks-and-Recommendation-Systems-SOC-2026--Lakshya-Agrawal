import matplotlib.pyplot as plt

# 1. Added the Baseline test to the front of the list
experiments = ['Baseline\n(10 Epochs)', 'Extended\n(25 Epochs)', 'Dim = 32\n(10 Epochs)', 'LR = 0.005\n(10 Epochs)', 'Batch = 1024\n(10 Epochs)']

# 2. Added the Baseline RMSE (0.9263) to the front of the scores
rmse_scores = [0.9263, 0.9230, 0.9228, 0.9363, 0.9255]

# Create the Bar Chart
plt.figure(figsize=(11, 6)) # Widened slightly for 5 bars

# 3. Added a 5th color ('#ef4444' - red) to the front for the baseline
bars = plt.bar(experiments, rmse_scores, color=['#ef4444', '#6366f1', '#06b6d4', '#10b981', '#f59e0b'])

# Add labels and title
plt.ylabel('Test RMSE (Lower is Better)')
plt.title('GraphSAGE Performance Across Hyperparameter Tests')
plt.ylim(0.91, 0.95) # Zoomed in to highlight the microscopic differences

# Add actual values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, round(yval, 4), ha='center', va='bottom', fontweight='bold')

# Save the graph as an image
plt.savefig('hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
print("Graph saved successfully as hyperparameter_comparison.png!")