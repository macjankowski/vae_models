
import util
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli


class DenseLayer:

    def __init__(self, in_dim, out_dim, f=tf.nn.relu):
        self.f = f
        self.W = tf.Variable(tf.truncated_normal(shape=(in_dim, out_dim), stddev=0.1), name="W")
        self.b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="bias")

    def forward(self, x):
        act = self.f(tf.matmul(x, self.W) + self.b)
        return act


class VeAutoencoder:

    def __init__(self, input_dim, hidden_dim):

        self.encoder_layers = []

        self.X = tf.placeholder(tf.float32, shape=(None, input_dim))

        # encoder
        in_size = input_dim
        for out_size in hidden_dim[:-1]:
            h = DenseLayer(in_size, out_size)
            self.encoder_layers.append(h)
            in_size = out_size

        last_dim = hidden_dim[-1]
        h = DenseLayer(in_size, 2 * last_dim, f=lambda x: x)
        self.encoder_layers.append(h)

        current_value = self.X
        for layer in self.encoder_layers:
            current_value = layer.forward(current_value)

        self.means = current_value[:, :last_dim]
        self.stdevs = tf.nn.softplus(current_value[:, last_dim:]) + 1e-6

        n = Normal(
            loc=self.means,
            scale=self.stdevs,
        )

        self.Z = n.sample()

        # decoder
        self.decoder_layers = []

        in_size = last_dim
        for out_size in reversed(hidden_dim[:-1]):
            h = DenseLayer(in_size, out_size)
            self.decoder_layers.append(h)
            in_size = out_size

        h = DenseLayer(in_size, input_dim, f=lambda x: x)
        self.decoder_layers.append(h)

        current_value = self.Z
        for layer in self.decoder_layers:
            current_value = layer.forward(current_value)

        logits = current_value
        posterior_predictive_logits = logits
        self.X_hat_distribution = Bernoulli(logits)
        self.posterior_predictive = self.X_hat_distribution.sample()
        self.posterior_predictive_probs = tf.nn.sigmoid(logits)

        expected_log_likelihood = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.X,
            logits=posterior_predictive_logits
        )
        expected_log_likelihood = tf.reduce_sum(expected_log_likelihood, 1)

        kl = -tf.log(self.stdevs) + 0.5 * (self.stdevs ** 2 + self.means ** 2) - 0.5
        kl = tf.reduce_sum(kl, axis=1)

        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-self.elbo)

    def fit(self, X, epochs, batch_sz):
        costs = []
        n_batches = len(X) // batch_sz
        print("n_batches:", n_batches)

        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            for i in range(epochs):
                np.random.shuffle(X)
                for j in range(n_batches):
                    batch = X[j * batch_sz:(j + 1) * batch_sz]
                    _, c = sess.run((self.train_op, self.elbo), feed_dict={self.X: batch})
                    c /= batch_sz
                    costs.append(c)
                    if j % 100 == 0:
                        print("iter: %d, cost: %.3f" % (j, c))
            plt.plot(costs)
            plt.show()


def main():
    X, y = util.get_mnist()
    autoencoder = VeAutoencoder(784,[10,2])
    autoencoder.fit(X,100, 64)


if __name__ == '__main__':
    main()

