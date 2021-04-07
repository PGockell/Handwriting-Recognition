import numpy as np
import joblib
from matplotlib import pyplot as plt

# quick plotting of posted image data

data = joblib.load('image.pkl')
image = np.zeros((28, 28))
for i in range(28):
    for j in range(28):
        image[i, j] = data[i * 28 + j]
        
plt.imshow(image, interpolation='none',
           cmap=plt.cm.get_cmap("Greys"))
plt.show()
