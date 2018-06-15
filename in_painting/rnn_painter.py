
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import csv
import pandas
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

import extract_data

DATA_DIR = "./data/"
SINGLE_PIXEL_FILENAME = DATA_DIR + 'one_pixel_inpainting.npy'
MULTI_PIXEL_FILENAME = DATA_DIR + '2X2_pixels_inpainting.npy'

METRICS_FOLDER = "../models/task2/metrics/"
MODELS_FOLDER = "../models/task2/"
MODEL_FILENAMES = ['T2_mod_32.ckpt', 'T2_mod_64.ckpt', 'T2_mod_128.ckpt', 'T2_mod_s32.ckpt']
METRICS_FILENAMES = ['T2_met_32.npy', 'T2_met_64.npy', 'T2_met_128.npy', 'T2_met_s32.npy']

# Parameters
INPUT_SIZE = 1      # One pixel at a time
STEPS = 784         # Number of steps
NUM_CLASSES = 10    # 10 output classes

# ==================================================================================================
# CHOOSE RUN OPTIONS ---- CHANGE STUFF HERE
# ==================================================================================================

EXTRACT_MNIST = False       # Set True if the pickle mnist file doesn't already exist
RUN_TEST = True            # Set True to run pixel-prediction (might take up to 10 minutes)
RUN_METRICS = False          # Set True to run loss metrics on pre-existing test data
SAVE_IMAGES = False          # Set True if you want to save a new batch of images at ../images/
MODEL_NUM = 0               # Choose which model to run options (0,1,2,3) for (34,64,128,s32)
PRED_SIZE = 1               # Choose the number of pixels to predict (0,1,2,3) for (1,10,28,300)
# ==================================================================================================


class RNN:
    def __init__(self, batch_size=500, num_hidden=32, num_linear=100, num_layers=1):

        # Initialize hyper-parameters
        self.batch_size = batch_size
        self.num_hid_units = num_hidden
        self.num_lin_units = num_linear
        self.num_hid_layers = num_layers

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
        if self.num_hid_layers > 1:
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


    def run(self, model_filename, orgb_images, ppds_images):

        y_ns, y_pred = self.model()

        im_store = np.zeros((1000, 784))
        pred_ns = np.zeros((1000, 784))

        # Launch test session
        with tf.Session() as sess:

            tf.global_variables_initializer().run()

            # load model
            is_successful = load_model(sess, model_filename)

            if not is_successful:
                return -1

            mpx_idx = np.where(ppds_images == ppds_images.min())
            temp = []

            for im in range(ppds_images.shape[0]):

                store = np.zeros(2)
                min_id = mpx_idx[im]

                for i in [0, 1]:

                    ppds_images[im, min_id] = i

                    images = np.reshape(ppds_images, (1000, 784, 1))
                    pixel_ns, pixel_prob = sess.run([y_ns, y_pred], feed_dict={self.x: images})
                    # Loss wrt. predicted image
                    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pixel_ns,
                                                                                 images), 1)
                    loss = sess.run(loss)
                    store[i] = loss
                    temp.append(pixel_ns)

                if store[0] < store[1]:
                    ppds_images[im, min_id] = 0
                    im_store[im] = ppds_images[im, :]
                    pred_ns[im] = temp[0]
                else:
                    ppds_images[im, min_id] = 1
                    im_store[im] = ppds_images[im, :]
                    pred_ns[im] = temp[0]


            pixel_ns = np.concatenate((np.zeros((1000, 1)), pixel_ns), axis=1)

            # Add ground truth images for later reference
            gt_images = np.repeat(orgb_images, 10, axis=0)
            gt_images = np.reshape(gt_images, (100, 10, 784))

            # Save Results
            if self.num_hid_layers > 1:
                FILENAME = METRICS_FOLDER + "T2_met_s" + str(self.num_hid_units)
            else:
                FILENAME = METRICS_FOLDER + "T2_met_" + str(self.num_hid_units)

            out_data = np.array([gt_images, pred_images, pred_ns])
            np.save(FILENAME, out_data)


        def metrics(self, filename, pred_size):
            """
            Computes the cross-entropy loss between for pixel prediction task
            :return:
            """

            # Load data from .npy file
            data = np.load(filename)

            # Raw loss over each pixel
            gt_images = np.reshape(data[0], [1000, 784])
            pred_images = np.reshape(data[1], [1000, 784])
            pred_ns = np.reshape(data[2], [1000, 784])

            # Loss wrt. predicted image
            ip_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred_ns, pred_images),
                                    1)

            # Loss wrt GT
            gt_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred_ns, gt_images), 1)

            # Launch test session
            with tf.Session() as sess:

                ip_loss = sess.run(ip_loss)
                gt_loss = sess.run(gt_loss)

                ip_loss = np.reshape(ip_loss, [100, 10])
                gt_loss = np.reshape(gt_loss, [100, 10])

                ip_loss_mean = np.mean(ip_loss, axis=1)
                gt_loss_mean = np.mean(gt_loss, axis=1)

            # Save Losses for each image
            if self.num_hid_layers > 1:
                FILENAME = METRICS_FOLDER + "T2_loss_s" + str(self.num_hid_units) + '_pn_' + str(
                    pred_size)
            else:
                FILENAME = METRICS_FOLDER + "T2_loss_" + str(self.num_hid_units) + '_pn_' + str(
                    pred_size)

            sample_loss = np.array([ip_loss, gt_loss])
            np.save(FILENAME, sample_loss)

            print('In-painting Xen, GT Xen: \n')
            print(np.column_stack((ip_loss_mean, gt_loss_mean)))

            # Write loss comparison to CSV
            with open(FILENAME + '.csv', 'w') as csvfile:
                csv_writer = csv.writer(csvfile)
                for ln in range(len(ip_loss_mean)):
                    csv_writer.writerow([ip_loss_mean[ln], gt_loss_mean[ln]])

        def save_bin_images(self, filename, loss_filename, pred_size):

            FAIL_FOLDER_NAME = '../images/' + str(self.num_hid_units) + '/' + str(pred_size) + '/'
            GOOD_FOLDER_NAME = '../images/' + str(self.num_hid_units) + '/' + str(pred_size) + '/'
            VAR_FOLDER_NAME = '../images/' + str(self.num_hid_units) + '/' + str(pred_size) + '/'

            if self.num_hid_layers > 1:
                FAIL_FOLDER_NAME = '../images/Stacked GRU32/' + str(pred_size) + '/'
                GOOD_FOLDER_NAME = '../images/Stacked GRU32/' + str(pred_size) + '/'
                VAR_FOLDER_NAME = '../images/Stacked GRU32/' + str(pred_size) + '/'

            # Load data from .npy file
            data = np.load(filename)
            loss_data = np.load(loss_filename)

            # Randomly choose 5 images
            image_idx = np.random.choice(100, 5)

            imgs = np.reshape(data[1], [1000, 784])[:, :483 + pred_size]

            # Only save the predicted part a set the rest to zero
            imgs = np.concatenate((imgs, np.zeros((1000, 300 - pred_size + 1))), axis=1)
            imgs = np.reshape(imgs, [100, 10, 784])

            for image_id in image_idx:
                sm_max = np.argmax(loss_data[1][image_id])
                sm_min = np.argmin(loss_data[1][image_id])
                sm_mid = 4

                # Save failure example
                img_filename = FAIL_FOLDER_NAME + 'I' + str(image_id + 1) + 'FAIL_sm_' + str(
                    sm_max + 1) + \
                               '.png'
                plt.imsave(img_filename, np.array(imgs[image_id, sm_max, :]).reshape(28, 28))

                # Save good example
                img_filename = GOOD_FOLDER_NAME + 'I' + str(image_id + 1) + 'GOOD_sm_' + str(
                    sm_min + 1) + \
                               '.png'
                plt.imsave(img_filename, np.array(imgs[image_id, sm_min, :]).reshape(28, 28))

                # Save high variance example
                img_filename = VAR_FOLDER_NAME + 'I' + str(image_id + 1) + 'VAR_sm' + str(
                    sm_mid + 1) + \
                               '.png'
                plt.imsave(img_filename, np.array(imgs[image_id, sm_mid, :]).reshape(28, 28))

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


def main():

    mod_hid_units = [32, 64, 128, 32]
    pred_size = [1, 10, 28, 300]
    num_layers = [1, 1, 1, 3]

    data = np.load(SINGLE_PIXEL_FILENAME)

    # Images with pixels missing
    ppds_images = data[0]
    # Ground-truth images
    orgb_images = data[1]

    rnn_obj = RNN(num_hidden=mod_hid_units[MODEL_NUM], num_layers=num_layers[MODEL_NUM])

    if RUN_TEST:
        model_filename = MODELS_FOLDER + MODEL_FILENAMES[MODEL_NUM]
        rnn_obj.run(model_filename, orgb_images, ppds_images)

    if RUN_METRICS:
        metrics_filename = METRICS_FOLDER + METRICS_FILENAMES[MODEL_NUM]
        rnn_obj.metrics(metrics_filename, pred_size[PRED_SIZE])

    if SAVE_IMAGES:
        metrics_filename = METRICS_FOLDER + METRICS_FILENAMES[MODEL_NUM]

        if num_layers[MODEL_NUM] > 1:
            loss_filename = METRICS_FOLDER + "T2_loss_s" + str(mod_hid_units[MODEL_NUM]) + '_pn_' + \
                                                         str(pred_size[PRED_SIZE]) + '.npy'
        else:
            loss_filename = METRICS_FOLDER + "T2_loss_" + str(mod_hid_units[MODEL_NUM]) + '_pn_' + \
                                                         str(pred_size[PRED_SIZE]) + '.npy'

        rnn_obj.save_bin_images(metrics_filename, loss_filename, pred_size[PRED_SIZE])

if __name__ == '__main__':
    main()
