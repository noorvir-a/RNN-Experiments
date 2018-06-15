from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pickle
import numpy as np


EXTRACT_BIN_MNIST = False
EXTRACT_PRED_IMAGES = True

def sample_images():
    """
    Returns the specified number of images from the prediction dataset
    :return:
    """
    orgb_images, ppds_images = load_mnist(dataset_type='PRED')

    ppds_images = ppds_images.reshape(100, 784, 1)
    orgb_images = orgb_images.reshape(100, 784, 1)

    return orgb_images, ppds_images


def binarize(images, threshold=0.1):
    """
    :param images:
    :param threshold:
    :return:
    """
    return (threshold < images).astype('float32')


def load_mnist(dataset_type):
    """
    Loads either the binarised MNIST dataset or the pixel-prediction dataset
    :return:
    """
    if dataset_type == 'MNIST':

        # load binarised MNIST data
        with open('mnist_bin.pickle', 'rb') as f:
            print("\nLoading MNIST data from pickle file.\n")
            train_x, train_y, val_x, val_y, test_x, test_y = pickle.load(f)
            data = {
                'train_x': np.array(train_x),
                'train_y': np.array(train_y),
                'val_x': np.array(val_x),
                'val_y': np.array(val_y),
                'test_x': np.array(test_x),
                'test_y': np.array(test_y)
            }
            return data

    elif dataset_type == 'PRED':
        # load pixel-prediction MNIST data-set
        with open('mnist_pred.pickle', 'rb') as f:
            print("\nLoading MNIST pixel-prediction image data-setfrom pickle "
                  "file.\n")
            orgb_images, ppds_images = pickle.load(f)

        return orgb_images, ppds_images


def create_prediction_dataset():
    """
    Removes the bottom 300 pixels of the input image and sets them to zero
    :return:
    """
    bin_mnist = load_mnist(dataset_type='MNIST')
    images = bin_mnist['test_x']
    orgb_images = np.copy(images)

    images[:, -300:] = np.zeros([1, 300])
    ppds_images = images

    # Randomly choose 100 image samples
    idx = np.random.choice(ppds_images.shape[0], 100, replace=False)
    # Change image batch into a form that can be fed into the trained model [num_images, 784, 1]
    ppds_images = ppds_images[idx]
    orgb_images = orgb_images[idx]

    # Store both the original binary test images as well as images with the last 300 pixels
    # removed(ppds_images)
    with open('mnist_pred.pickle', 'wb') as f:
        pickle.dump([orgb_images,
                     ppds_images], f, protocol=2)

def extract_mnist():
    """
    Use TensorFlow to extract MNIST data
    :return:
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print('Converting MNIST images to binary...\n')
    train_X = binarize(mnist.train.images[:])
    train_Y = binarize(mnist.train.labels[:])
    val_X = binarize(mnist.validation.images[:])
    val_Y = binarize(mnist.validation.labels[:])
    test_X = binarize(mnist.test.images[:])
    test_Y = binarize(mnist.test.labels[:])

    with open('mnist_bin.pickle', 'wb') as f:
        pickle.dump([train_X,
                     train_Y,
                     val_X,
                     val_Y,
                     test_X,
                     test_Y], f, protocol=2)


if __name__ == '__main__':

    if EXTRACT_BIN_MNIST:
        extract_mnist()

    if EXTRACT_PRED_IMAGES:
        create_prediction_dataset()
