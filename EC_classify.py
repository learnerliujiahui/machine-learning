#using utf-8 coding
# data = 3.27
__author__ = 'Liu Jiahui'

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math

def main():

    image_size = 28
    num_channels = 1
    pixel_depth = 255
    num_labels = 10
    validation_size = 5000  # Size of the validation set.
    random_seed = 66478  # Set to None for random seed.
    batch_size = 64
    num_epoches = 10
    EVAL_batch_size = 64
    EVAL_frequency = 100  # Number of steps between evaluations.

    mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    # data shape can be adjusted
    # None means that the dimention can be of any length
    X = tf.placeholder(tf.float32,[None,image_size,image_size,1])
    # total 4 categories, this is the label
    Y_ = tf.placeholder(tf.float32,[None,num_labels])
    # variable learning rate
    lr = tf.placeholder(tf.float32)

    K = 4  # first convolutional layer output depth
    L = 8  # second convolutional layer output depth
    M = 12  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.ones([K])/10)
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L])/10)
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M])/10)
    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N])/10)
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.ones([10])/10)

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    Ylogits = tf.matmul(Y4, W5) + B5
    # the result of the predict
    Y = tf.nn.softmax(Ylogits)

    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training step, the learning rate is a placeholder
    # Optimizer that implements the Adam algorithm
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    # init
    init = tf.global_variables_initializer()

    #------------------------------------------------------------
    sess = tf.Session()
    sess.run(init)

    # can be initialized by the numpy model
    # batch_X ="input data"
    # batch_Y ="input label"
    for i in range(1000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate})
    print(accuracy[0])
if __name__=="__main__":
    main()