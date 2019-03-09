import numpy as np
import tensorflow as tf

sample = np.array([[1.947,-0.259,4.27,1.407,1.567,-8.1,4.799,6.377,0.986,5.626,3.079,2.888,8.719,-1.977,
                    9.307,3.986,-4.16,0.328,5.469,-5.848,1.063,-0.131,3.341,1.402,0.932,3.219,9.717,
                    -4.186,7.054,0.045,-1.563,-0.564,-3.918,-3.784,2.112,-3.258,2.3,-8.386,3.762,5.717,
                    1.913,-2.107,-1.703,4.817,-4.855,-4.634,5.482,-6.946,8.021,-1.085,2.578,-2.349,
                    -3.449,1.628,-3.872,5.726,-5.694,8.034,-4.169,-6.812,-2.363,-0.22,4.384,1.79,3.464,
                    8.232,16.486,12.031,0.958,3.851,-0.898,-1.787,-2.549,1.772,-6.307,2.684,-10.575,
                    5.415,2.721,-2.981,0.704,-0.705,6.792,3.695,2.793,8.488,-2.355,-2.625,-1.312,2.186,
                    4.72,4.954,8.599,-7.744,1.521,-1.445,2.701,11.404,1.047,0.309,-6.19,1.405,-2.038,
                    9.793,5.581,8.222,7.214,-7.551,8.932,3.185,1.276,-1.368,-2.567,-3.317,10.643,-0.542,
                    4.856,0.148,13.388,0.734,7.004,-0.266,5.519,10.749,-1.6,1.876,-2.553,0.422,10.083,
                    -4.423,3.031,1.647,5.494,0.952,3.551,3.862,-3.679,-0.125,7.362,-1.147,5.143,5.986,
                    4.871,-4.628,-4.068,4.532,-4.099,5.639,2.254,-0.091,4.806,-0.306,1.012,6.812,-1.012,
                    1.627,7.799,3.497,6.22,0.095,-4.748,3.979,3.137,7.14,-0.696,2.678,4.564,-0.795,4.194,
                    17.029,-3.14,4.834,11.941,3.882,-3.871,10.001,-8.004,1.581,3.91,8.11,-8.216,3.184,
                    5.841,-3.512,6.621,1.714,11.368,8.698,4.902,-1.459,2.23,-1.121,3.474,-5.301,6.035,
                    5.286,4.725,3.146,5.632,6.358]]).T

mean = tf.Variable(0, dtype=tf.float64)
stdev = tf.Variable(1, dtype=tf.float64)

x = tf.placeholder(tf.float64)

dist = tf.contrib.distributions.Normal(loc=mean, scale=stdev)

log_likelihood_arr = dist.log_prob(x)
log_likelihood = tf.reduce_sum(log_likelihood_arr)

optimizer = tf.train.GradientDescentOptimizer(0.005)
#optimizer = tf.train.RMSPropOptimizer(0.5)
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
train_op = optimizer.minimize(-1.0 * log_likelihood)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_op, feed_dict={x: sample})
    if i % 100 == 0:
        print(sess.run(log_likelihood, feed_dict={x: sample}))


sample_mean = np.sum(sample)/len(sample)
sample_variance = np.sum(np.power(sample - sample_mean,2))/(len(sample) - 1)
#print("Analitical estimation from sample")
print(sample_mean)
print(np.sqrt(sample_variance))

#print("Estimation from optimization")
print(sess.run(mean))
print(sess.run(stdev))
