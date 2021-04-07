"""Creating a ML model for handwriting recognition using
the MNIST data set.

First we have to decode the original file format (IDX)
and construct the dataset by matching images and labels.

Then we can learn a simple model with sklearn.
"""

from collections import namedtuple
import struct
import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets


def load_images(images_path, n_images) -> np.ndarray:
    """Loads image data from the IDX file format.

    As the images are in a low level file format
    we first have to decode the images.
    Data from http://yann.lecun.com/exdb/mnist/

    Args:
       images_path (str) : Path to the dataset
       n_images (int) : Number of images to be loaded

    Returns:
       (List[np.ndarray]) : Images as numpy 2D-array from the dataset
    """
    ds_images = []
    with open(images_path, "rb") as f:
        # the head of the file is encoded in 32-bit integers in big-edian
        magic_number_32bit = f.read(4)
        images_number_32bit = f.read(4)
        rows_number_32bit = f.read(4)
        columns_numbers_32bit = f.read(4)

        _ = struct.unpack('>i', magic_number_32bit)[0]
        images_number = struct.unpack('>i', images_number_32bit)[0]
        rows_number = struct.unpack('>i', rows_number_32bit)[0]
        columns_number = struct.unpack('>i', columns_numbers_32bit)[0]

        # TODO:
        # make this more performant for 60000 images
        # with 28x28 pixels we loop 47,040,000 times
        for _ in range(n_images):
            image = np.zeros((rows_number, columns_number))
            for i in range(rows_number):
                for j in range(columns_number):
                    byte = f.read(1)
                    pixel = struct.unpack('B', byte)[0]
                    image[i, j] = pixel
            ds_images.append(image)
    return ds_images


def load_labels(labels_path, n_images):
    labels = []
    with open(labels_path, "rb") as f:
        magic_number_32bit = f.read(4)
        labels_number_32bit = f.read(4)

        _ = struct.unpack('>i', magic_number_32bit)[0]
        labels_number = struct.unpack('>i', labels_number_32bit)[0]
        for _ in range(n_images):
            byte = f.read(1)
            label = struct.unpack('B', byte)[0]
            labels.append(label)
    return labels

def load_dataset(images_path, labels_path, n_images):
    Dataset = namedtuple('Dataset', 'images labels')
    images = load_images(images_path, n_images)
    labels = load_labels(labels_path, n_images)
    return Dataset(images, labels)

def plot_data(dataset):
    fig = plt.figure()
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        plt.gca().set_title(f'{dataset.labels[i]}')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(dataset.images[i], interpolation='none',
                   cmap=plt.cm.get_cmap("Greys"))
    plt.show()


if __name__ == "__main__":
    TRAINING_IMAGES = 20000
    TEST_IMAGES = 100
    dataset_training = load_dataset("data/train-images-idx3-ubyte",
                                    "data/train-labels-idx1-ubyte",
                                    TRAINING_IMAGES)

    dataset_test = load_dataset("data/t10k-images-idx3-ubyte",
                                "data/t10k-labels-idx1-ubyte",
                                TEST_IMAGES)

    # K-NN algorithm from scikit-learn
    X = np.array([x.flatten() for x in dataset_training.images])
    y = np.array(dataset_training.labels)
    X_test = np.array([x.flatten() for x in dataset_test.images])
    y_test = np.array(dataset_test.labels)

    n_neighbors = 5
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X, y)

    # save current model
    joblib_file = "model.pkl"
    joblib.dump(clf, joblib_file)

    score = clf.score(X_test, y_test)
    print(f'score with {n_neighbors}-NN: {score}')

    image = dataset_test[0][0]
    plt.imshow(image, interpolation='none',
                   cmap=plt.cm.get_cmap("Greys"))
    plt.show()
    # Z = clf.predict(X_test)
    # predictions = list(zip(images_test, Z))
    # plot_data(predictions)
    # plot_data(dataset_test)
