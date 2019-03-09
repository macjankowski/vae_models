import numpy as np
import tensorflow as tf

sample = np.array([[1.,0.,1.,1.,1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,
                    0.,1.,1.,1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.,
                    1.,1.,1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.,1.,
                    1.,1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.,1.,1.,
                    1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,0.,1.,1.,1.,
                    0.,0.,1.,0.,0.,1.,0.,0.,1.,1.,1.,1.,1.,1.,
                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]]).T

np.random.shuffle(sample)

theta = tf.Variable(0.5)
x = tf.placeholder(tf.float32)

dist = tf.contrib.distributions.Bernoulli(probs=theta)

log_likelihood = tf.reduce_sum(dist.log_prob(x))

optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(-1.0 * log_likelihood)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train_op, feed_dict={x: sample})
    print(sess.run(theta))
    print(sess.run(log_likelihood, feed_dict={x: sample}))

print(sess.run(theta))
print(np.mean(sample))