import joblib
from matplotlib import pyplot as plt

# quick plotting of posted image data

image = joblib.load('
plt.imshow(image, interpolation='none',
           cmap=plt.cm.get_cmap("Greys"))
plt.show()
