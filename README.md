## RNN-Experiments

This repository contains code for some of my early experiments with Recurrent Neural Networks (RNNs) - LSTMs and GRUs. The experiments are applied on image data ([MNIST](http://yann.lecun.com/exdb/mnist/))- since its easier to understand intuitively. They include:

* [Classification](https://github.com/noorvir-a/RNN-Experiments/tree/master/classification)
* [Pixel-prediction](https://github.com/noorvir-a/RNN-Experiments/tree/master/pixel_prediction)
* [In-painting](https://github.com/noorvir-a/RNN-Experiments/tree/master/in_painting)


I recommend using this code only for reference; I wrote it a while ago for personal use only.

### Sample Results

**Classification**

The goal here is to classify images from the MNIST data-set into one of ten categories by exploiting the sequential memory of RNNs. Instead of treating the image as a 2D arrangement of pixels, we treat it as a 1D vector. The objective is to learn to predict the joint probability of the pixels in an image.

**Pixel Prediction**

Here we try and predict the next pixel `(i + 1)` in a sequence based on the current pixel `(i)` and the recurrent state of the network.

 Completing images in this way can have interesting results. Below is an example of a case where I removed the bottom half of the image and  completed the rest one pixel at a time. Although the original image had the label "one", the network completed it as a "six", resulting in high cross-entropy loss over the entire image. This is obviously a little unfair since even a human could be expected to complete the image as a "six" if he were only to see the top half of the ground-truth image. This highlights the need carefully to pick a loss function that's suitable for the task at hand.

Predicted completion:

![](https://github.com/noorvir-a/RNN-Experiments/tree/master/images/pixel_completion_ex1.png)

Ground-truth image:

![](https://github.com/noorvir-a/RNN-Experiments/tree/master/images/pixel_completion_gt1.png)


**In-painting**

This is similar to pixel-prediction but instead of predicting the next pixel in the sequence, we predict a *missing* pixel instead.
