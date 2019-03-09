import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

y_train = y_train.reshape(len(y_train), 1)
onehot_encoder_train = OneHotEncoder(sparse=False, categories='auto')
y_train = onehot_encoder_train.fit_transform(y_train)

y_test = y_test.reshape(len(y_test), 1)
onehot_encoder_test = OneHotEncoder(sparse=False, categories='auto')
y_test = onehot_encoder_test.fit_transform(y_test)


NUM_STEPS = 10
MINIBATCH_SIZE = 64
n_batches = len(X_train) // MINIBATCH_SIZE

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(NUM_STEPS):
        print("epoch: %d" % i)
        np.random.shuffle(X_train)
        for j in range(n_batches):
            batch_xs = X_train[j * MINIBATCH_SIZE:(j + 1) * MINIBATCH_SIZE]
            batch_ys = y_train[j * MINIBATCH_SIZE:(j + 1) * MINIBATCH_SIZE]

            sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y_true: y_train})
    print("Train Accuracy: {:.4}%".format(train_accuracy * 100))
    test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_true: y_test})
    print("Test Accuracy: {:.4}%".format(test_accuracy * 100))



