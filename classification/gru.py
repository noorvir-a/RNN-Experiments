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

MODEL_FOLDER = "../models/task1/"
LOG_FOLDER = "../training_logs/"

# ==================================================================================================
# CHOOSE RUN OPTIONS ---- CHANGE STUFF HERE
# ==================================================================================================

isTRAIN = True             # Set True to train and False to test
EXTRACT_MNIST = True       # Set True if the pickle mnist file doesn't already exist
NUM_HIDDEN_UNITS = 32

# Uncomment to run the respective models:
LOAD_MODEL_FILENAME = "./T1_b_32.ckpt"
# LOAD_MODEL_FILENAME = "./T1_b_64.ckpt"
# LOAD_MODEL_FILENAME = "./T1_b_128.ckpt"
# ==================================================================================================


class RNN:
    def __init__(self, batch_size=200, learning_rate=0.001, num_epochs=40, num_linear=100,
                 num_hidden=NUM_HIDDEN_UNITS):
        # Initialize hyper-parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_hid_units = num_hidden
        self.num_lin_units = num_linear

        # Initialize model variables and parameters
        self.x = tf.placeholder("float", [None, STEPS, INPUT_SIZE])
        self.y = tf.placeholder("float", [None, NUM_CLASSES])
        self.do = tf.placeholder("float", [None, 1])

        # Weights and biases
        self.weights_H2L = tf.Variable(tf.random_normal([self.num_hid_units,
                                                         self.num_lin_units]))
        self.weights_L2O = tf.Variable(tf.random_normal([self.num_lin_units,
                                                         NUM_CLASSES]))
        self.bias_H2L = tf.Variable(tf.random_normal([self.num_lin_units]))
        self.bias_L2O = tf.Variable(tf.random_normal([NUM_CLASSES]))

    def model(self):
        """
        Builds the Tensorflow graph
        :return:
        """
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
        outputs, state = rnn.rnn(gru, x, dtype=tf.float32)

        # First affine-transformation - output from last input
        y1 = tf.matmul(outputs[-1], self.weights_H2L) + self.bias_H2L
        y2 = tf.nn.relu(y1)
        y_pred = tf.matmul(y2, self.weights_L2O) + self.bias_L2O

        return y_pred


    def run(self, data, is_train):
        """
        Runs training and testing
        :param data:
        :param is_train:
        :return:
        """
        t = time.time()
        ts = datetime.datetime.fromtimestamp(t).strftime(
            '%Y%m%d -%H%M-%S')
        FILENAME = 'T1_b_' + str(self.num_hid_units) + '_' + ts

        MODEL_FILENAME = MODEL_FOLDER + FILENAME

        # Dimensions of data
        (NUM_TRAIN_SAMPLES, sizeX) = data['train_x'].shape
        (NUM_TEST_SAMPLES, sizeXT) = data['test_x'].shape

        acc_train = np.zeros((self.num_epochs * int(NUM_TRAIN_SAMPLES/self.batch_size), 1))
        acc_test = np.zeros((self.num_epochs * int(NUM_TRAIN_SAMPLES / self.batch_size), 1))
        c_graph = self.model()

        print('Doing a bit more graph-building\n')
        # Accuracy measure
        prediction_outcome = tf.equal(tf.argmax(c_graph, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction_outcome, tf.float32))

        # Loss-function
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_graph,
                                                                         labels=self.y))

        if is_train:
            trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(ce_loss)

            # Launch training session
            with tf.Session() as sess:

                tf.global_variables_initializer().run()

                # Overall iteration count
                it = 0
                st_time = time.time()

                print("Starting training for %d epochs... \n" % self.num_epochs)

                # Output CSV
                csv_filename = LOG_FOLDER + 'T1_b_' + str(self.num_hid_units) + '_' + ts + '.csv'

                with open(csv_filename, 'w') as csvfile:

                    # write CSV header and other meta info
                    csv_meta = ['batch_size = ' + str(self.batch_size),
                                'learning_rate = ' + str(self.learning_rate),
                                'num_hidden_units = ' + str(self.num_hid_units)]
                    csv_header = ['epoch', 'batch_number', 'hrs', 'mins',
                                  'seconds', 'train_acc', 'test_acc', 'loss']

                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(csv_meta)
                    csv_writer.writerow(csv_header)

                    print("Epoch,batch_number,hrs,mins,seconds,"
                          "train_acc, test_acc, loss\n")

                    # Run optimisation over mini-batches
                    for epoch in range(self.num_epochs):

                        # Initialise mini-batch start and end indices
                        mb_st = 0
                        mb_en = mb_st + self.batch_size

                        for batch in range(int(NUM_TRAIN_SAMPLES / self.batch_size)):
                            batch_x = data['train_x'][mb_st:mb_en]
                            batch_x = batch_x.reshape((self.batch_size, STEPS, INPUT_SIZE))
                            batch_y = data['train_y'][mb_st:mb_en]

                            _, loss = sess.run([trainer, ce_loss], feed_dict={
                                                            self.x: batch_x,
                                                            self.y: batch_y})

                            # Print the accuracy on every 10th mini-batch iteration
                            if (it + 1) % 100 == 0:
                                # Calculate accuracy on training set
                                btrain_x = data['train_x'][:]
                                btrain_x = btrain_x.reshape(
                                    (NUM_TRAIN_SAMPLES, STEPS, INPUT_SIZE))
                                btrain_y = data['train_y'][:]

                                # Calculate accuracy on test set
                                btest_x = data['test_x'][:]
                                btest_x = btest_x.reshape((NUM_TEST_SAMPLES, STEPS, INPUT_SIZE))
                                btest_y = data['test_y'][:]

                                acc_train[it] = sess.run(accuracy, feed_dict={self.x: btrain_x,
                                                                              self.y: btrain_y})
                                acc_test[it] = sess.run(accuracy, feed_dict={self.x: btest_x,
                                                                             self.y: btest_y})

                                el_time = time.time() - st_time
                                m, sec = divmod(el_time, 60)
                                hr, m = divmod(m, 60)

                                csv_writer.writerow([epoch, batch + 1, hr, m, int(sec),
                                                     str(acc_train[it]), str(acc_train[it]),
                                                     str(loss)])
                                print("%d,%d,%d,%d,%d,%.4f,%.4f,%.4f\n" % (epoch, batch + 1, hr,
                                                                           m, sec, acc_train[it],
                                                                           acc_test[it], loss))

                                if (it + 1) % 1000 == 0:
                                    # Save model continuously
                                    save_model(sess, MODEL_FILENAME)

                            mb_st = mb_en
                            mb_en = mb_st + self.batch_size
                            it += 1

                save_model(sess, MODEL_FILENAME)

        else:
            # Launch test session
            with tf.Session() as sess:

                tf.global_variables_initializer().run()

                # load model
                load_model(sess, MODEL_FOLDER + LOAD_MODEL_FILENAME)

                # Test Data
                btest_x = data['test_x'][:]
                btest_x = btest_x.reshape((NUM_TEST_SAMPLES, STEPS, INPUT_SIZE))
                btest_y = data['test_y'][:]

                # Test Accuracy
                acc_test, loss = sess.run([accuracy, ce_loss], feed_dict={self.x: btest_x,
                                                                          self.y: btest_y})
                print('Test Accuracy: %.5f, Test Loss: %.5f \n' % (acc_test, loss))


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
                     test_Y], f)


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

