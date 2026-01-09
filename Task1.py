import numpy as np
import matplotlib.pyplot as plt

# Data points extracted from the graph
x = np.array([-9.9, -7.4, -5.1, -2.5, -0.1, 2.2, 4.8, 7.3, 9.7])
y = np.array([-7.3, -5.6, -3.2, -1.5, 0.6, 2.9, 4.6, 6.8, 8.4])

# Pearson correlation coefficient
r = np.corrcoef(x, y)[0, 1]

print("Pearson correlation coefficient (r):", r)

# Scatter plot
plt.figure()
plt.scatter(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title(f"Scatter Plot of Data Points (r = {r:.2f})")
plt.show()
