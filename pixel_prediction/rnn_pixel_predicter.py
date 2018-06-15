from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import csv
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

DATA_DIR = "./MNIST_data/"

# Parameters
INPUT_SIZE = 1      # One pixel at a time
STEPS = 784         # Number of steps
NUM_CLASSES = 10    # 10 output classes


MODEL_FOLDER = "../models/task2/"
LOG_FOLDER = "../training_logs/task2/"
LOAD_MODEL_FILENAME = "../models/task1/" + "T1_a_32.ckpt"

# ==================================================================================================
# CHOOSE RUN OPTIONS ---- CHANGE STUFF HERE
# ==================================================================================================

isTRAIN = True             # Set True to train
EXTRACT_MNIST = False       # Set True if the pickle mnist file doesn't already exist
# ==================================================================================================

class RNN:
    def __init__(self, batch_size=300, learning_rate=0.0015, num_epochs=130,
                 num_hidden=32, num_linear=100, num_hid_layers=3):
        # Initialize hyper-parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_hid_units = num_hidden
        self.num_lin_units = num_linear
        self.num_hid_layers = num_hid_layers

        # Initialize model variables and parameters
        self.x = tf.placeholder("float", [None, STEPS, INPUT_SIZE])
        # The ground-truth value in this case is over the number of pixels
        # minus one
        self.y = tf.placeholder("float", [None, STEPS-1])
        # Weights and biases
        self.weights_H2O = tf.Variable(tf.random_normal([self.num_hid_units,
                                                         1]))
        self.bias_H2O = tf.Variable(tf.random_normal([1]))

    def model(self):

        print('Building model\n')
        # We don't want to modify to original tensor
        x = self.x
        # Reshape input into a list of tensors of the correct size
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, INPUT_SIZE])
        # Since we're using one pixel at a time, transform list of vector of
        # 784x1
        x = tf.split(0, STEPS, x)

        # Define LSTM cells and get outputs list and states
        gru = rnn_cell.GRUCell(self.num_hid_units)
        gru = rnn_cell.DropoutWrapper(gru, output_keep_prob=1)
        gru = rnn_cell.MultiRNNCell([gru] * self.num_hid_layers)
        outputs, state = rnn.rnn(gru, x, dtype=tf.float32)

        # Turn result back into [batch_size, steps, hidden_units] format.
        outputs = tf.transpose(outputs, [1, 0, 2])
        # Flatten into [batch_size x steps, hidden_units] to allow matrix
        # multiplication
        outputs = tf.reshape(outputs, [-1, self.num_hid_units])

        # Apply affine transformation to reshape output [batch_size x steps, 1]
        y1 = tf.matmul(outputs, self.weights_H2O) + self.bias_H2O
        y1 = tf.reshape(y1, [-1, STEPS])
        # Keep prediction (sigmoid applied) and non-sigmoid (apply sigmoid in
        #  cost function)
        y_ns = y1[:, :783]
        y_pred = tf.sigmoid(y1)[:, :783]

        return y_ns, y_pred


    def run(self, data, is_train):
        """
        Runs training and testing
        :param data:
        :param is_train:
        :return:
        """
        t = time.time()
        ts = datetime.datetime.fromtimestamp(t).strftime(
            '%Y%m%d%H%M%S')
        FILENAME = 'T2_b_' + str(self.num_hid_units) + '_' + ts

        MODEL_FILENAME = MODEL_FOLDER + FILENAME

        # Dimensions of data
        (NUM_TRAIN_SAMPLES, sizeX) = data['train_x'].shape
        (NUM_TEST_SAMPLES, sizeXT) = data['test_x'].shape

        acc_train = np.zeros((self.num_epochs * int(NUM_TRAIN_SAMPLES/self.batch_size), 1))
        y_ns, y_pred = self.model()

        print('Doing a bit more graph-building\n')

        if is_train:
            # Loss-function
            ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y_ns, self.y), 1)
            ce_loss = tf.reduce_mean(ce_loss)
            trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(ce_loss)

            # Launch training session
            with tf.Session() as sess:

                tf.global_variables_initializer().run()

                # Overall iteration count
                it = 0
                st_time = time.time()

                print("Starting training for %d epochs... \n" % self.num_epochs)

                # Output CSV
                csv_filename = LOG_FOLDER + 'T2_b' + str(self.num_hid_units) + '_' + ts + '.csv'

                with open(csv_filename, 'w') as csvfile:

                    # write CSV header and other meta info
                    csv_meta = ['batch_size = ' + str(self.batch_size),
                                'learning_rate = ' + str(self.learning_rate),
                                'num_hidden_units = ' + str(self.num_hid_units)]
                    csv_header = ['epoch', 'batch_number', 'hrs', 'mins',
                                  'seconds', 'accuracy', 'loss']

                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(csv_meta)
                    csv_writer.writerow(csv_header)

                    print("Epoch,batch_number,hrs,mins,seconds,accuracy\n")

                    save_model(sess, MODEL_FILENAME)

                    # Run optimisation over mini-batches
                    for epoch in range(self.num_epochs):

                        # Initialise mini-batch start and end indices
                        mb_st = 0
                        mb_en = mb_st + self.batch_size

                        for batch in range(int(NUM_TRAIN_SAMPLES / self.batch_size)):
                            batch_x = data['train_x'][mb_st:mb_en]
                            batch_x = batch_x.reshape((self.batch_size, STEPS, INPUT_SIZE))

                            batch_y = data['train_x'][mb_st:mb_en]
                            # Remove the first pixel of each image
                            batch_y = batch_y[:, 1:]

                            _, loss = sess.run([trainer, ce_loss], feed_dict={self.x: batch_x,
                                                                              self.y: batch_y})

                            # Print the accuracy on every 10th mini-batch iteration
                            if (it + 1) % 100 == 0:

                                el_time = time.time() - st_time
                                m, sec = divmod(el_time, 60)
                                hr, m = divmod(m, 60)

                                csv_writer.writerow([epoch, batch+1, hr, m, int(sec),
                                                     str(acc_train[it]), str(loss)])
                                print("%d,%d,%d,%d,%d,%.4f,%.4f\n"
                                      % (epoch, batch+1, hr, m, sec, acc_train[
                                        it], loss))

                                if (it + 1) % 1000 == 0:
                                    # Save model continuously
                                    save_model(sess, MODEL_FILENAME)

                            mb_st = mb_en
                            mb_en = mb_st + self.batch_size
                            it += 1

                save_model(sess, MODEL_FILENAME)


def save_model(sess, filename):
    """
        :param sess:
        :param filename:
        :return:
    """
    if not os.path.exists(MODEL_FOLDER):
        print('Creating path where to save model: ' + MODEL_FOLDER)

        os.mkdir(MODEL_FOLDER)

    print('Saving model at: ' + filename)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.save(sess, filename)
    print('Model successfully saved.\n')


def load_model(sess, filename):
    """
    :param sess:
    :param filename:
    :return:
    """
    if os.path.exists(filename):
        print('\nLoading save model from: ' + filename)
        saver = tf.train.Saver()
        saver.restore(sess, filename)
        print('Model succesfully loaded.\n')
        return True
    else:
        print('Model file <<' + filename + '>> does not exists!')
        return False


def binarize(images, threshold=0.1):
    """
    :param images:
    :param threshold:
    :return:
    """
    return (threshold < images).astype('float32')


def extract_data():
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


def main():

    # Extract MNIST data from TensorFlow
    if EXTRACT_MNIST:
        print("Extracting data\n")
        extract_data()

    # load data from pre-extracted pickle file
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

    rnn_obj = RNN()

    if isTRAIN:
        rnn_obj.run(data, is_train=True)
    else:
        rnn_obj.run(data, is_train=False)


if __name__ == '__main__':
    main()