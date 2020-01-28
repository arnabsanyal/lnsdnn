###################################################
# 
# Author - Arnab Sanyal
# USC. Spring 2019
###################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import argparse

def softmax(inp):

	# inp_shape = inp.shape
	max_vals = np.max(inp, axis=1)
	max_vals.shape = max_vals.size, 1
	u = np.exp(inp - max_vals)
	v = np.sum(u, axis=1)
	v.shape = v.size, 1
	u = u / v
	return u

def main(main_params):

	is_training = bool(main_params['is_training'])
	leaking_coeff = float(main_params['leaking_coeff'])
	batchsize = int(main_params['minibatch_size'])
	lr = float(main_params['learning_rate'])
	num_epoch = int(main_params['num_epoch'])
	_lambda = float(main_params['lambda'])
	ones = np.ones((batchsize, 1))
		
	if is_training:
		# load mnist data and split into train and test sets
		# one-hot encoded target column

		file = np.load('./../../datasets/mnist.npz', 'r') # dataset
		x_train = file['train_data']
		y_train = file['train_labels']
		x_test = file['test_data']
		y_test = file['test_labels']
		x_train, y_train = shuffle(x_train, y_train)
		x_test, y_test = shuffle(x_test, y_test)
		file.close()

		split = int(main_params['split'])
		x_val = x_train[split:]
		y_val = y_train[split:]
		y_train = y_train[:split]
		x_train = x_train[:split]

		# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

		W1 = np.random.normal(0, 0.1, (785, 100))
		W2 = np.random.normal(0, 0.1, (101, 10))

		delta_W1 = np.zeros(W1.shape)
		delta_W2 = np.zeros(W2.shape)

		performance = {}
		performance['loss_train'] = np.zeros(num_epoch)
		performance['acc_train'] = np.zeros(num_epoch)
		performance['acc_val'] = np.zeros(num_epoch)

		accuracy = 0.0

		for epoch in range(num_epoch):
			print('At Epoch %d:' % (1 + epoch))
			loss = 0.0
			for mbatch in range(int(split / batchsize)):

				start = mbatch * batchsize
				x = x_train[start:(start + batchsize)]
				y = y_train[start:(start + batchsize)]

				s1 = np.hstack((ones, x)) @ W1
				###################################################
				mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
				###################################################
				a1 = s1 * mask
				s2 = np.hstack((ones, a1)) @ W2
				a2 = softmax(s2)

				cat_cross_ent = np.log(a2) * y
				cat_cross_ent[np.isnan(cat_cross_ent)] = 0
				loss -= np.sum(cat_cross_ent)

				grad_s2 = (a2 - y) / batchsize
				###################################################
				delta_W2 = np.hstack((ones, a1)).T @ grad_s2
				###################################################
				grad_a1 = grad_s2 @ W2[1:].T
				grad_s1 = mask * grad_a1
				################################################### 
				delta_W1 = np.hstack((ones, x)).T @ grad_s1
				###################################################
				# grad_x =

				W2 -= (lr * (delta_W2 + (_lambda * W2)))
				W1 -= (lr * (delta_W1 + (_lambda * W1)))

			loss /= split
			performance['loss_train'][epoch] = loss
			print('Loss at epoch %d: %f' %((1 + epoch), loss))
			correct_count = 0
			for mbatch in range(int(split / batchsize)):

				start = mbatch * batchsize
				x = x_train[start:(start + batchsize)]
				y = y_train[start:(start + batchsize)]

				s1 = np.hstack((ones, x)) @ W1
				###################################################
				mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
				###################################################
				a1 = s1 * mask
				s2 = np.hstack((ones, a1)) @ W2

				correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

			accuracy = correct_count / split
			performance['acc_train'][epoch] = 100 * accuracy
			print("Train-set accuracy at epoch %d: %f" % ((1 + epoch), performance['acc_train'][epoch]))

			correct_count = 0
			for mbatch in range(int(x_val.shape[0] / batchsize)):

				start = mbatch * batchsize
				x = x_val[start:(start + batchsize)]
				y = y_val[start:(start + batchsize)]

				s1 = np.hstack((ones, x)) @ W1
				###################################################
				mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
				###################################################
				a1 = s1 * mask
				s2 = np.hstack((ones, a1)) @ W2

				correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

			accuracy = correct_count / x_val.shape[0]
			performance['acc_val'][epoch] = 100 * accuracy
			print("Val-set accuracy at epoch %d: %f\n" % ((1 + epoch), performance['acc_val'][epoch]))

		correct_count = 0
		for mbatch in range(int(x_test.shape[0] / batchsize)):

			start = mbatch * batchsize
			x = x_test[start:(start + batchsize)]
			y = y_test[start:(start + batchsize)]

			s1 = np.hstack((ones, x)) @ W1
			###################################################
			mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
			###################################################
			a1 = s1 * mask
			s2 = np.hstack((ones, a1)) @ W2

			correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

		accuracy = 100.0 * (correct_count / x_test.shape[0])
		print('Test-set performance: %f' % accuracy)

		np.savez_compressed('./lin_model_MNIST.npz', W1=W1, W2=W2, loss_train=performance['loss_train'], \
			acc_train=performance['acc_train'], acc_val=performance['acc_val'])

	else:

		file = np.load('./lin_model_MNIST.npz', 'r')
		W1 = file['W1']
		W2 = file['W2']
		performance = {}
		performance['loss_train'] = file['loss_train']
		performance['acc_train'] = file['acc_train']
		performance['acc_val'] = file['acc_val']
		file.close()

		file = np.load('./../../datasets/mnist.npz', 'r') # dataset
		x_test = file['test_data']
		y_test = file['test_labels']
		x_test, y_test = shuffle(x_test, y_test)
		file.close()

		correct_count = 0
		for mbatch in range(int(x_test.shape[0] / batchsize)):
			
			start = mbatch * batchsize
			x = x_test[start:(start + batchsize)]
			y = y_test[start:(start + batchsize)]

			s1 = np.hstack((ones, x)) @ W1
			###################################################
			mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
			###################################################
			a1 = s1 * mask
			s2 = np.hstack((ones, a1)) @ W2

			correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

		accuracy = 100.0 * (correct_count / x_test.shape[0])
		print('Test-set performance: %f' % accuracy)

	'''
	The model architecture that we trained is as follows 
		_________________________________________________________________
		
			OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####         784
          InputLayer     |   -------------------         0     0.0%
                       #####         784
               Dense   XXXXX -------------------     78500    98.7%
          Leaky relu   #####         100
               Dense   XXXXX -------------------      1010     1.3%
             softmax   #####          10
		=================================================================
		Total params: 79,510
		Trainable params: 79,510
		Non-trainable params: 0
		_________________________________________________________________
	'''

	# Plots for training accuracies

	if is_training:
		
		fig = plt.figure(figsize = (16, 9)) 
		ax = fig.add_subplot(111)
		x = range(1, 1 + performance['loss_train'].size)
		ax.plot(x, performance['acc_train'], 'r')
		ax.plot(x, performance['acc_val'], 'g')
		ax.set_xlabel('Number of Epochs')
		ax.set_ylabel('Accuracy')
		ax.set_title('Test-set Accuracy at %.2f%%' % accuracy)
		plt.suptitle('Validation and Training Accuracies', fontsize=14)
		ax.legend(['train', 'validation'])
		plt.grid(which='both', axis='both', linestyle='-.')

		plt.savefig('accuracy.png')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--is_training', default = False)
	parser.add_argument('--split', default = 50000)
	parser.add_argument('--learning_rate', default = 0.01)
	parser.add_argument('--lambda', default = 0.001)
	parser.add_argument('--minibatch_size', default = 5)
	parser.add_argument('--num_epoch', default = 20)
	parser.add_argument('--leaking_coeff', default = 0.0078125)
	args = parser.parse_args()
	main_params = vars(args)
	main(main_params)