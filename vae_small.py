from __future__ import print_function, division
from builtins import range, input

import util

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli


class DenseLayer:

    def __init__(self, in_dim, out_dim, f=tf.nn.relu, name="dense"):
        self.name = name
        with tf.name_scope(self.name):
            self.f = f
            self.W = tf.Variable(tf.truncated_normal(shape=(in_dim, out_dim), stddev=0.1), name="W")
            self.b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="bias")
            tf.summary.histogram("weights", self.W)
            tf.summary.histogram("bias", self.b)

    def forward(self, X):
        with tf.name_scope(self.name):
            act = self.f(tf.matmul(X, self.W) + self.b)
            tf.summary.histogram("activation", act)
            return act


class VeAutoencoderSmall:

    def encode(self, X, input_dim, hidden_dims):
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims[:-1]:
            h = DenseLayer(in_dim, h_dim)
            encoder_layers.append(h)
            in_dim = h_dim

        middle_layer_dim = hidden_dims[-1]
        encoder_layers.append(DenseLayer(in_dim, 2 * middle_layer_dim, f=lambda x: x))

        current_value = X
        for layer in encoder_layers:
            current_value = layer.forward(current_value)

        means = current_value[:, :middle_layer_dim]
        stdevs = tf.nn.softplus(current_value[:, middle_layer_dim:]) + 1e-6
        return means, stdevs

    def decode(self, Z, input_dim, hidden_dims):
        decoder_layers = []

        in_dim = hidden_dims[-1]
        for hidden_dim in reversed(hidden_dims[:-1]):
            h = DenseLayer(in_dim, hidden_dim)
            decoder_layers.append(h)
            in_dim = hidden_dim

        decoder_layers.append(DenseLayer(in_dim, input_dim, f=lambda x: x))

        current_value = Z
        for decoder_layer in decoder_layers:
            current_value = decoder_layer.forward(current_value)

        return current_value

    def __init__(self, input_dim, hidden_dims):
        self.X = tf.placeholder(tf.float32, shape=(None, input_dim))

        #with tf.name_scope('input_reshape'):
        #    image_shaped_input = tf.reshape(self.X, [-1, 28, 28, 1])
        #    tf.summary.image('input', image_shaped_input, 10)

        #encoder
        means, stdevs = self.encode(self.X, input_dim, hidden_dims)

        n = Normal(
          loc=means,
          scale=stdevs,
        )
        Z = n.sample()

        #decoder
        self.logits = self.decode(Z, input_dim, hidden_dims)

        self.X_hat_distribution = Bernoulli(logits=self.logits)
        self.posterior_predictive = self.X_hat_distribution.sample()

        #with tf.name_scope('sample_output_reshaped'):
        #    posterior_predictive_reshaped = tf.reshape(self.posterior_predictive, [-1, 28, 28, 1])
        #    tf.summary.image('sample_output', tf.cast(posterior_predictive_reshaped, tf.float32), 10)

        self.posterior_predictive_probs = tf.nn.sigmoid(self.logits)

        with tf.name_scope('probs_output_reshaped'):
            posterior_predictive_probs_reshaped = tf.reshape(self.posterior_predictive_probs, [-1, 28, 28, 1])
            tf.summary.image('probs_output', posterior_predictive_probs_reshaped, 10)

        expected_log_likelihood = tf.reduce_sum(
            self.X_hat_distribution.log_prob(self.X),
            1
        )
        #tf.summary.scalar("Expected log-likelihood", tf.reduce_sum(expected_log_likelihood))

        self.kl = -tf.log(stdevs) + 0.5 * (stdevs ** 2 + means ** 2) - 0.5
        self.kl = tf.reduce_sum(self.kl, axis=1)
        tf.summary.scalar("KL", tf.reduce_sum(self.kl))

        self.elbo = tf.reduce_sum(expected_log_likelihood - self.kl)
        #tf.summary.scalar("ELBO", self.elbo/64)

        #self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-self.elbo)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

        self.merged_summary = tf.summary.merge_all()

        self.writer_train = tf.summary.FileWriter("/tmp/vae/deep/13/train")
        self.writer_train.add_graph(self.sess.graph)

    def fit(self, X, epochs=30, batch_sz=64):
        costs=[]
        n_batches = len(X) // batch_sz
        print("n_batches:", n_batches)

        iter = 1
        for i in range(epochs):
            print("epoch: %d" % i)
            np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz:(j + 1) * batch_sz]
                _, c = self.sess.run((self.train_op, self.elbo), feed_dict={self.X: batch})
                c /= batch_sz
                costs.append(-c)
                if j % 100 == 0:
                    s = self.sess.run(self.merged_summary, feed_dict={self.X: batch})
                    self.writer_train.add_summary(s, iter)
                    print("iter: %d, cost: %.3f" % (j, c))
                iter += 1

        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return self.sess.run(self.posterior_predictive, feed_dict={self.X: X})

    def predict_probs(self, X):
        return self.sess.run(self.posterior_predictive_probs, feed_dict={self.X: X})


def main():
    print('Starting autoencoder')
    X,y = util.get_mnist()

    #X = (X > 0.5).astype(np.float32)

    model = VeAutoencoderSmall(784, [300, 100])
    model.fit(X, epochs=10)

    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = model.predict_probs([x]).reshape(28,28)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28,28), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title('Reconstruction')

        ans = input('Generate another?')
        if ans and ans[0] in ('n' or 'N'):
            done = True


if __name__ == '__main__':
    main()
