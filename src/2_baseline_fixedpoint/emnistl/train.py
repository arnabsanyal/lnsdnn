###################################################
# 
# Author - Arnab Sanyal
# USC. Spring 2019
###################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import argparse

class fixed_point:
    def __init__(self, qi, qf):
        self.lshifter = (1 << qf)
        self.rshifter = (2 ** -qf)
        self.intlim = (1 << (qi - 1))
        self.maxfrac = 1 - self.rshifter

    def quantize_array(self, _matrix_m1):
        whole = _matrix_m1.astype('int64')
        frac = _matrix_m1 - whole
        frac = frac * self.lshifter
        frac = np.round(frac)
        frac = frac * self.rshifter
        f1 = (whole >= self.intlim)
        f2 = (whole < -self.intlim)
        whole[f1] = (self.intlim - 1)
        whole[f2] = -self.intlim
        frac[f1] = self.maxfrac
        frac[f2] = -self.maxfrac
        whole = whole + frac
        whole[_matrix_m1 == -np.inf] = -np.inf
        whole[_matrix_m1 == np.inf] = np.inf
        return whole

    def quantize_array_p(self, _matrix_m1):
        f1 = (np.floor(_matrix_m1) >= self.intlim)
        f2 = (np.floor(_matrix_m1) < -self.intlim)
        if(np.sum(f1)):
            print(_matrix_m1[f1])
            exit(0)
        if(np.sum(f2)):
            print(_matrix_m1[f2])
            exit(0)
        return _matrix_m1

def softmax(inp):

    max_vals = np.max(inp, axis=1)
    max_vals.shape = max_vals.size, 1
    u = np.exp(inp - max_vals)
    v = np.sum(u, axis=1)
    v.shape = v.size, 1
    u = u / v
    return u

def main(main_params):

    is_training = bool(main_params['is_training'])
    qi = int(main_params['bi'])
    qf = int(main_params['bf'])
    leaking_coeff = float(main_params['leaking_coeff'])
    batchsize = int(main_params['minibatch_size'])
    lr = float(main_params['learning_rate'])
    _lambda = float(main_params['lambda'])
    num_epoch = int(main_params['num_epoch'])
    ones = np.ones((batchsize, 1))
    fp = fixed_point(qi, qf)
    _step = 10
    print('lambda: %f' %_lambda)
    print('bi: %d\tbf: %d' % (qi, qf))
        
    if is_training:
        # load mnist data and split into train and test sets
        # one-hot encoded target column

        file = np.load('./../../datasets/emnist_letters.npz', 'r') # dataset
        x_train = fp.quantize_array(file['train_data'])
        y_train = file['train_labels']
        x_test = fp.quantize_array(file['test_data'])
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

        W1 = fp.quantize_array(np.random.normal(0, 0.1, (785, 100)))
        W2 = fp.quantize_array(np.random.normal(0, 0.1, (101, 26)))

        delta_W1 = np.zeros(W1.shape)
        delta_W2 = np.zeros(W2.shape)

        performance = {}
        performance['loss_train'] = np.zeros(num_epoch)
        performance['acc_train'] = np.zeros(num_epoch)
        performance['acc_val'] = np.zeros(num_epoch)

        accuracy = 0.0

        for epoch in range(num_epoch):
            print('At Epoch %d:' % (1 + epoch))
            # if (epoch % _step == 0) and (epoch != 0):
            #     lr = lr * 0.1
            loss = 0.0
            for mbatch in range(int(split / batchsize)):

                start = mbatch * batchsize
                x = x_train[start:(start + batchsize)]
                y = y_train[start:(start + batchsize)]

                s1 = fp.quantize_array(np.hstack((ones, x)) @ W1)
                ###################################################
                mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
                ###################################################
                a1 = fp.quantize_array(s1 * mask)
                s2 = fp.quantize_array(np.hstack((ones, a1)) @ W2)
                a2 = softmax(s2)

                cat_cross_ent = np.log(a2) * y
                cat_cross_ent[np.isnan(cat_cross_ent)] = 0
                loss -= np.sum(cat_cross_ent)

                grad_s2 = fp.quantize_array_p((a2 - y) / batchsize)
                ###################################################
                delta_W2 = fp.quantize_array_p(np.hstack((ones, a1)).T @ grad_s2)
                ###################################################
                grad_a1 = fp.quantize_array_p(grad_s2 @ W2[1:].T)
                grad_s1 = fp.quantize_array_p(mask * grad_a1)
                ################################################### 
                delta_W1 = fp.quantize_array_p(np.hstack((ones, x)).T @ grad_s1)
                ###################################################
                # grad_x =

                # W2 -= fp.quantize_array(lr * (delta_W2 + (_lambda * W2)))
                # W1 -= fp.quantize_array(lr * (delta_W1 + (_lambda * W1)))
                W2 = fp.quantize_array_p(W2 - (lr * (delta_W2 + (_lambda * W2))))
                W1 = fp.quantize_array_p(W1 - (lr * (delta_W1 + (_lambda * W1))))

            loss /= split
            performance['loss_train'][epoch] = loss
            print('Loss at epoch %d: %f' %((1 + epoch), loss))
            correct_count = 0
            for mbatch in range(int(split / batchsize)):

                start = mbatch * batchsize
                x = x_train[start:(start + batchsize)]
                y = y_train[start:(start + batchsize)]

                s1 = fp.quantize_array(np.hstack((ones, x)) @ W1)
                ###################################################
                mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
                ###################################################
                a1 = fp.quantize_array(s1 * mask)
                s2 = fp.quantize_array(np.hstack((ones, a1)) @ W2)

                correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

            accuracy = correct_count / split
            performance['acc_train'][epoch] = 100 * accuracy
            print("Train-set accuracy at epoch %d: %f" % ((1 + epoch), performance['acc_train'][epoch]))

            correct_count = 0
            for mbatch in range(int(x_val.shape[0] / batchsize)):

                start = mbatch * batchsize
                x = x_val[start:(start + batchsize)]
                y = y_val[start:(start + batchsize)]

                s1 = fp.quantize_array(np.hstack((ones, x)) @ W1)
                ###################################################
                mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
                ###################################################
                a1 = fp.quantize_array(s1 * mask)
                s2 = fp.quantize_array(np.hstack((ones, a1)) @ W2)

                correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

            accuracy = correct_count / x_val.shape[0]
            performance['acc_val'][epoch] = 100 * accuracy
            print("Val-set accuracy at epoch %d: %f\n" % ((1 + epoch), performance['acc_val'][epoch]))

        correct_count = 0
        for mbatch in range(int(x_test.shape[0] / batchsize)):

            start = mbatch * batchsize
            x = x_test[start:(start + batchsize)]
            y = y_test[start:(start + batchsize)]

            s1 = fp.quantize_array(np.hstack((ones, x)) @ W1)
            ###################################################
            mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
            ###################################################
            a1 = fp.quantize_array(s1 * mask)
            s2 = fp.quantize_array(np.hstack((ones, a1)) @ W2)

            correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

        accuracy = 100.0 * (correct_count / x_test.shape[0])
        print('Test-set performance: %f' % accuracy)

        np.savez_compressed('./lin_model_EMNIST_letters_1_%d_%d.npz' %(qi - 1, qf), W1=W1, W2=W2, loss_train=performance['loss_train'], \
            acc_train=performance['acc_train'], acc_val=performance['acc_val'])

    else:

        file = np.load('./lin_model_EMNIST_letters_1_%d_%d.npz' %(qi - 1, qf), 'r')
        W1 = file['W1']
        W2 = file['W2']
        performance = {}
        performance['loss_train'] = file['loss_train']
        performance['acc_train'] = file['acc_train']
        performance['acc_val'] = file['acc_val']
        file.close()

        file = np.load('./../../datasets/emnist_letters.npz', 'r') # dataset
        x_test = fp.quantize_array(file['test_data'])
        y_test = file['test_labels']
        x_test, y_test = shuffle(x_test, y_test)
        file.close()

        correct_count = 0
        for mbatch in range(int(x_test.shape[0] / batchsize)):
            
            start = mbatch * batchsize
            x = x_test[start:(start + batchsize)]
            y = y_test[start:(start + batchsize)]

            s1 = fp.quantize_array(np.hstack((ones, x)) @ W1)
            ###################################################
            mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
            ###################################################
            a1 = fp.quantize_array(s1 * mask)
            s2 = fp.quantize_array(np.hstack((ones, a1)) @ W2)

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
               Dense   XXXXX -------------------     78500    96.8%
          Leaky relu   #####         100
               Dense   XXXXX -------------------      2626     3.2%
             softmax   #####          26
        =================================================================
        Total params: 81,126
        Trainable params: 81,126
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
    parser.add_argument('--split', default = 104000)
    parser.add_argument('--learning_rate', default = 0.015625)
    parser.add_argument('--minibatch_size', default = 5)
    parser.add_argument('--num_epoch', default = 20)
    parser.add_argument('--lambda', default = 0.0009765625)
    parser.add_argument('--leaking_coeff', default = 0.0078125)
    parser.add_argument('--bi', default = 6)
    parser.add_argument('--bf', default = 18)
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)