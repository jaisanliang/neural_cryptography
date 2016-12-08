import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
	sess = tf.Session()

	# Hyperparameters
	BATCH_SIZE = 512
	N = 16					# number of bits in message
	K = 16					# number of bits in shared symmetric key
	learning_rate = 0.0008

	train_key_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, K))
	train_message_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, N))

	l1_num_hidden = N + K
	l1_depth = 1
	l1_ab_stddev = 1.0/np.sqrt(N+K)
	l1_e_stddev = 1.0/np.sqrt(N)

	l2_filter_size = 4
	l2_depth = 2
	l2_stride = 1
	l2_stddev = 1.0/np.sqrt(l2_filter_size**2*l1_depth)

	l3_filter_size = 2
	l3_depth = 4
	l3_stride = 2
	l3_stddev = 1.0/np.sqrt(l3_filter_size**2*l2_depth)

	l4_filter_size = 1
	l4_depth = 4
	l4_stride = 1
	l4_stddev = 1.0/np.sqrt(l4_filter_size**2*l3_depth)

	l5_filter_size = 1
	l5_depth = 1
	l5_stride = 1
	l5_stddev = 1.0/np.sqrt(l5_filter_size**2*l4_depth)

	# Alice's network weights/parameters
	a_l1_weights = tf.Variable(tf.truncated_normal(
		[N+K, l1_num_hidden], stddev=l1_ab_stddev), name='a_l1_weights')
	a_l1_biases = tf.Variable(tf.zeros([l1_num_hidden]), name='a_l1_biases')

	a_l2_weights = tf.Variable(tf.truncated_normal(
		[l2_filter_size, l1_depth, l2_depth], stddev=l2_stddev), name='a_l2_weights')
	a_l2_biases = tf.Variable(tf.zeros([l2_depth]), name='a_l2_biases')

	a_l3_weights = tf.Variable(tf.truncated_normal(
		[l3_filter_size, l2_depth, l3_depth], stddev=l3_stddev), name='a_l3_weights')
	a_l3_biases = tf.Variable(tf.zeros([l3_depth]), name='a_l3_biases')

	a_l4_weights = tf.Variable(tf.truncated_normal(
		[l4_filter_size, l3_depth, l4_depth], stddev=l4_stddev), name='a_l4_weights')
	a_l4_biases = tf.Variable(tf.zeros([l4_depth]), name='a_l4_biases')

	a_l5_weights = tf.Variable(tf.truncated_normal(
		[l5_filter_size, l4_depth, l5_depth], stddev=l5_stddev), name='a_l5_weights')
	a_l5_biases = tf.Variable(tf.zeros([l5_depth]), name='a_l5_biases')

	# Bob's network weights/parameters
	b_l1_weights = tf.Variable(tf.truncated_normal(
		[N+K, l1_num_hidden], stddev=l1_ab_stddev), name='b_l1_weights')
	b_l1_biases = tf.Variable(tf.zeros([l1_num_hidden]), name='b_l1_biases')

	b_l2_weights = tf.Variable(tf.truncated_normal(
		[l2_filter_size, l1_depth, l2_depth], stddev=l2_stddev), name='b_l2_weights')
	b_l2_biases = tf.Variable(tf.zeros([l2_depth]), name='b_l2_biases')

	b_l3_weights = tf.Variable(tf.truncated_normal(
		[l3_filter_size, l2_depth, l3_depth], stddev=l3_stddev), name='b_l3_weights')
	b_l3_biases = tf.Variable(tf.zeros([l3_depth]), name='b_l3_biases')

	b_l4_weights = tf.Variable(tf.truncated_normal(
		[l4_filter_size, l3_depth, l4_depth], stddev=l4_stddev), name='b_l4_weights')
	b_l4_biases = tf.Variable(tf.zeros([l4_depth]), name='b_l4_biases')

	b_l5_weights = tf.Variable(tf.truncated_normal(
		[l5_filter_size, l4_depth, l5_depth], stddev=l5_stddev), name='b_l5_weights')
	b_l5_biases = tf.Variable(tf.zeros([l5_depth]), name='b_l5_weights')

	# Eve's network weights/parameters
	e_l1_weights = tf.Variable(tf.truncated_normal(
		[N, l1_num_hidden], stddev=l1_e_stddev), name='e_l1_weights')
	e_l1_biases = tf.Variable(tf.zeros([l1_num_hidden]), name='e_l1_biases')

	e_l2_weights = tf.Variable(tf.truncated_normal(
		[l2_filter_size, l1_depth, l2_depth], stddev=l2_stddev), name='e_l2_weights')
	e_l2_biases = tf.Variable(tf.zeros([l2_depth]), name='e_l2_biases')

	e_l3_weights = tf.Variable(tf.truncated_normal(
		[l3_filter_size, l2_depth, l3_depth], stddev=l3_stddev), name='e_l3_weights')
	e_l3_biases = tf.Variable(tf.zeros([l3_depth]), name='e_l3_biases')

	e_l4_weights = tf.Variable(tf.truncated_normal(
		[l4_filter_size, l3_depth, l4_depth], stddev=l4_stddev), name='e_l4_weights')
	e_l4_biases = tf.Variable(tf.zeros([l4_depth]), name='e_l4_biases')

	e_l5_weights = tf.Variable(tf.truncated_normal(
		[l5_filter_size, l4_depth, l5_depth], stddev=l5_stddev), name='e_l5_weights')
	e_l5_biases = tf.Variable(tf.zeros([l5_depth]), name='e_l5_biases')

	def model(train = False):
		# Alice's layers
		concat = tf.concat(1,[train_key_node,train_message_node])
		a1 = tf.nn.sigmoid(tf.matmul(concat,a_l1_weights) + a_l1_biases)
		shape = a1.get_shape().as_list()
		a1 = tf.reshape(a1, [shape[0], shape[1], 1])

		a2 = tf.nn.conv1d(a1, a_l2_weights, l2_stride, padding='SAME')
		a2 = tf.nn.sigmoid(a2 + a_l2_biases)

		a3 = tf.nn.conv1d(a2, a_l3_weights, l3_stride, padding='SAME')
		a3 = tf.nn.sigmoid(a3 + a_l3_biases)

		a4 = tf.nn.conv1d(a3, a_l4_weights, l4_stride, padding='SAME')
		a4 = tf.nn.sigmoid(a4 + a_l4_biases)

		a5 = tf.nn.conv1d(a4, a_l5_weights, l5_stride, padding='SAME')
		a5 = tf.nn.tanh(a5 + a_l5_biases)
		shape = a5.get_shape().as_list()
		a5 = tf.reshape(a5, [shape[0], shape[1]])
		
		# Bob's layers
		concat = tf.concat(1,[train_key_node,a5])
		b1 = tf.nn.sigmoid(tf.matmul(concat,b_l1_weights) + b_l1_biases)
		shape = b1.get_shape().as_list()
		b1 = tf.reshape(b1, [shape[0], shape[1], 1])

		b2 = tf.nn.conv1d(b1, b_l2_weights, l2_stride, padding='SAME')
		b2 = tf.nn.sigmoid(b2 + b_l2_biases)

		b3 = tf.nn.conv1d(b2, b_l3_weights, l3_stride, padding='SAME')
		b3 = tf.nn.sigmoid(b3 + b_l3_biases)

		b4 = tf.nn.conv1d(b3, b_l4_weights, l4_stride, padding='SAME')
		b4 = tf.nn.sigmoid(b4 + b_l4_biases)

		b5 = tf.nn.conv1d(b4, b_l5_weights, l5_stride, padding='SAME')
		b5 = tf.nn.tanh(b5 + b_l5_biases)
		shape = b5.get_shape().as_list()
		b5 = tf.reshape(b5, [shape[0], shape[1]])
		
		# Eve's layers
		e1 = tf.nn.sigmoid(tf.matmul(a5,e_l1_weights) + e_l1_biases)
		shape = e1.get_shape().as_list()
		e1 = tf.reshape(e1, [shape[0], shape[1], 1])

		e2 = tf.nn.conv1d(e1, e_l2_weights, l2_stride, padding='SAME')
		e2 = tf.nn.sigmoid(e2 + e_l2_biases)

		e3 = tf.nn.conv1d(e2, e_l3_weights, l3_stride, padding='SAME')
		e3 = tf.nn.sigmoid(e3 + e_l3_biases)

		e4 = tf.nn.conv1d(e3, e_l4_weights, l4_stride, padding='SAME')
		e4 = tf.nn.sigmoid(e4 + e_l4_biases)

		e5 = tf.nn.conv1d(e4, e_l5_weights, l5_stride, padding='SAME')
		e5 = tf.nn.tanh(e5 + e_l5_biases)
		shape = e5.get_shape().as_list()
		e5 = tf.reshape(e5, [shape[0], shape[1]])
		return b5, e5

	# Training computation
	bob_decode, eve_decode = model()
	b_loss = tf.reduce_mean(tf.abs(bob_decode-train_message_node))/2.0
	e_loss = tf.reduce_mean(tf.abs(eve_decode-train_message_node))/2.0
	ab_loss = b_loss + (N/2.0-e_loss)**2/(N/2.0)**2
	
	optimizer = tf.train.AdamOptimizer(learning_rate)

	all_vars = tf.trainable_variables()
	ab_vars = [v for v in all_vars if (v.name[0] == 'a' or v.name[0] == 'b')]
	e_vars = [v for v in all_vars if v.name[0] == 'e']

	# Per step, train Alice and Bob for one batch, then train Eve for two batches
	ab_train = optimizer.minimize(ab_loss, var_list=ab_vars)
	e_train = optimizer.minimize(e_loss, var_list=e_vars)

	num_training_steps=20000
	
	init = tf.initialize_all_variables()
	sess.run(init)

	b_losses = [0 for i in range(num_training_steps/100)]
	e_losses = [0 for i in range(num_training_steps/100)]

	for step in xrange(num_training_steps):
		# Randomly initialize keys and messages
		messages = np.random.randint(2,size=(BATCH_SIZE, N))*2-1
		keys = np.random.randint(2,size=(BATCH_SIZE, K))*2-1
		feed_dict = {train_key_node: keys,
					 train_message_node: messages}
		if step % 2 == 0:
			_,loss_b = sess.run([ab_train,b_loss], feed_dict=feed_dict)
			if step % 100 == 0:
				b_losses[step/100] = loss_b
		else:
			_,_,loss_e = sess.run([e_train,e_train,e_loss], feed_dict=feed_dict)
			if step % 100 == 1:
				e_losses[step/100] = loss_e

	plt.plot(range(num_training_steps/100),b_losses,'-r',range(num_training_steps/100),e_losses,'-b')
	plt.show()

	# 20 minutes for 10000 iterations

if __name__ == '__main__':
	main()
