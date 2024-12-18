import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('calibrated_lines.csv')

# Create the plot
fig, ax = plt.subplots(figsize=(15, 8))

# List to store x and y positions of labels to detect overlap
label_positions = []

for index, row in df.iterrows():
    # Define initial offset
    offset = 0.3 * abs(row['c'])
    
    # Create the vertical line
    ax.vlines(x=row['freq(GHz)'], ymin=0, ymax=abs(row['c']), color='blue', lw=1.5)
    
    # Initialize flag to detect overlap
    y_position = abs(row['c']) + offset
    x_position = row['freq(GHz)']
    
    # Check for overlap with previously placed labels (in both x and y directions)
    while any(abs(y_position - prev_y) < 0.5 and abs(x_position - prev_x) < 0.05 * x_position for prev_x, prev_y in label_positions):
        # Increase the offset and recalculate the y_position if overlap is detected
        offset += 0.3
        y_position = abs(row['c']) + offset
    
    # Add the label at the adjusted y_position
    text_artist = ax.text(x_position, y_position, row['Line'],
                          rotation=0, verticalalignment='bottom', horizontalalignment='center',
                          fontsize=7, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    # Store the new x_position and y_position to prevent future overlap
    label_positions.append((x_position, y_position))

# Setting the x-axis to logarithmic scale
ax.set_xscale('log')
ax.set_ylim(0, 2 * max(abs(df['c'])))

# Setting labels and title
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Line Ratio (C parameter)')
ax.set_title('Emission Lines by Frequency and Line Ratio')

# Ensure all labels fit by adjusting layout
plt.tight_layout()

# Show plot
plt.show()