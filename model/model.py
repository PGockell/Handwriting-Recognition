import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets

def predict(data):
    # load the previously trained model
    model = joblib.load('model/model.pkl')

    # build image up and flatten to ensure correct rotation
    data = np.array(data)
    image = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            image[i, j] = data[j * 28 + i]
    image = image.flatten()

    # Normalize the pixels into the range [0, 255].
    normalized_image = (image - image.min()) / (image.max() - image.min())
    normalized_image *= 255

    # for debugging by looking at the image that is used for classifcation
    joblib.dump(normalized_image, "util/image.pkl")

    return model.predict([normalized_image])
