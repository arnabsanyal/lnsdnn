###################################################
# 
# Author - Arnab Sanyal
# USC. Spring 2019
###################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import argparse
import native_matr_mult_wrapper

def matrixToLogTensor(mattr_a):                             # Bug Free
    assert len(mattr_a.shape) == 2, "Input Matrix shape is non-standard"

    _logTensor_lt1 = np.zeros(mattr_a.size * 2)
    _logTensor_lt1.shape = 2, mattr_a.shape[0], mattr_a.shape[1]
    _logTensor_lt1[0, :] = mattr_a > 0
    _logTensor_lt1[1, :] = np.log2(mattr_a * ((2 * _logTensor_lt1[0, :]) - 1))

    return _logTensor_lt1

def logTensorToMatrix(_logTensor_lt1):                      # Bug Free
    assert _logTensor_lt1.shape[0] == 2, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt1.shape) == 3, "Log Tensor shape is non-standard"

    mattr_a = np.zeros(_logTensor_lt1[0, :].shape)
    mattr_a = (2 ** _logTensor_lt1[1, :]) * ((2 * _logTensor_lt1[0, :]) - 1)

    return mattr_a

def log_hstack(_logTensor_lt1, _logTensor_lt2):
    assert _logTensor_lt1.shape[0] == 2, "Log Tensor shape is non-standard"
    assert _logTensor_lt2.shape[0] == 2, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt1.shape) == 3, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt2.shape) == 3, "Log Tensor shape is non-standard"

    retval = [[], []]
    retval[0] = np.hstack((_logTensor_lt1[0], _logTensor_lt2[0]))
    retval[1] = np.hstack((_logTensor_lt1[1], _logTensor_lt2[1]))
    return np.array(retval)

# def logleaky(_logTensor_lt1, leaking_coefficient):
    # assert len(_logTensor_lt1.shape) == 3, "Log Tensor shape is non-standard"
    # assert _logTensor_lt1.shape[0] == 2, "Log Tensor shape is non-standard"

    # result = np.zeros(_logTensor_lt1.shape)
    # result[1, :][_logTensor_lt1[0, :] == 0] = leaking_coefficient
    # result[0, :][_logTensor_lt1[0, :] == 1] = 1
    # result[1, :][_logTensor_lt1[0, :] == 1] = 0
    # # result[1, :] = 0

    # return result

class approximate_addition:
    def __init__(self, table_size, granularity):
        self.N = table_size
        self.granularity = granularity
        x = np.array(range(0, (self.N * self.granularity) + 1), dtype = float)
        x = x / self.granularity
        u = 2 ** (-x)
        self.delP = np.log2(1 + u)
        self.delM = np.log2(1 - u)

    def addition(self, _logTensor_lt1, _logTensor_lt2):
        assert _logTensor_lt1.shape[0] == 2, "Log-Tensor shape is non-standard"
        assert _logTensor_lt2.shape[0] == 2, "Log-Tensor shape is non-standard"
        assert len(_logTensor_lt1.shape) == 3, "Log-Tensor shape is non-standard"
        assert len(_logTensor_lt2.shape) == 3, "Log-Tensor shape is non-standard"

        # if _logTensor_lt1.shape[1] > _logTensor_lt2.shape[1]:
        #     _logTensor_lt2 = self.broadcast(_logTensor_lt2, int(_logTensor_lt1.shape[1] / _logTensor_lt2.shape[1]) - 1, axis=1)
        
        assert _logTensor_lt1.shape == _logTensor_lt2.shape, "Non-conformable matrices for element wise addition"

        diff = np.abs(_logTensor_lt1[1, :] - _logTensor_lt2[1, :])
        samesign = (_logTensor_lt1[0, :] == _logTensor_lt2[0, :])
        whichSign = np.zeros(_logTensor_lt1[0, :].shape)
        whichSign[_logTensor_lt1[1, :] < _logTensor_lt2[1, :]] = 1

        result = np.zeros(_logTensor_lt1.shape)
        result[1, :] = np.maximum(_logTensor_lt1[1, :], _logTensor_lt2[1, :])
        result[0, :] = whichSign * _logTensor_lt2[0, :] + (1 - whichSign) * _logTensor_lt1[0, :]

        diff[diff > self.N] = self.N
        diff[np.isnan(diff)] = self.N
        diff = (diff * self.granularity).astype(int)
        r_neg_ad = ((~samesign) * self.delM[diff])
        r_neg_ad[np.isnan(r_neg_ad)] = 0
        result[1, :] = result[1, :] + (samesign * self.delP[diff]) + r_neg_ad

        return result

    def subtraction(self, _logTensor_lt1, _logTensor_lt2):
        assert _logTensor_lt1.shape[0] == 2, "Log-Tensor shape is non-standard"
        assert _logTensor_lt2.shape[0] == 2, "Log-Tensor shape is non-standard"
        assert len(_logTensor_lt1.shape) == 3, "Log-Tensor shape is non-standard"
        assert len(_logTensor_lt2.shape) == 3, "Log-Tensor shape is non-standard"
        assert _logTensor_lt1.shape == _logTensor_lt2.shape, "Non-conformable matrices for element wise subtraction"

        _logTensor_lt2[0, :] = 1 - _logTensor_lt2[0, :]

        diff = np.abs(_logTensor_lt1[1, :] - _logTensor_lt2[1, :])
        samesign = (_logTensor_lt1[0, :] == _logTensor_lt2[0, :])
        whichSign = np.zeros(_logTensor_lt1[0, :].shape)
        whichSign[_logTensor_lt1[1, :] < _logTensor_lt2[1, :]] = 1

        result = np.zeros(_logTensor_lt1.shape)
        result[1, :] = np.maximum(_logTensor_lt1[1, :], _logTensor_lt2[1, :])
        result[0, :] = whichSign * _logTensor_lt2[0, :] + (1 - whichSign) * _logTensor_lt1[0, :]

        diff[diff > self.N] = self.N
        diff[np.isnan(diff)] = self.N
        diff = (diff * self.granularity).astype(int)
        r_neg_ad = ((~samesign) * self.delM[diff])
        r_neg_ad[np.isnan(r_neg_ad)] = 0
        result[1, :] = result[1, :] + (samesign * self.delP[diff]) + r_neg_ad

        _logTensor_lt2[0, :] = 1 - _logTensor_lt2[0, :]
        return result

    # def logsum(self, _logTensor_lt1, axis=1):
    #     assert _logTensor_lt1.shape[0] == 2, "Log-Tensor shape is non-standard"
    #     assert len(_logTensor_lt1.shape) == 3, "Log-Tensor shape is non-standard"
    #     assert (axis < 2) and (axis >= 0), "Invalid axis provided"

    #     if _logTensor_lt1.shape[axis] == 1:
    #         return _logTensor_lt1

    #     if axis == 2:
    #         _logTensor_lt1 = _logTensor_lt1.transpose((0, 2, 1))

    #     idx = _logTensor_lt1.shape[1]
    #     a = np.zeros((2, 1, _logTensor_lt1.shape[2]))
    #     b = np.zeros((2, 1, _logTensor_lt1.shape[2]))
    #     a[0, :] = _logTensor_lt1[0, :][0]
    #     a[1, :] = _logTensor_lt1[1, :][0]
    #     b[0, :] = _logTensor_lt1[0, :][1]
    #     b[1, :] = _logTensor_lt1[1, :][1]
    #     result = self.addition(a, b)

    #     if idx > 2:
    #         for i in range(2 , idx):
    #             a[0, :] = _logTensor_lt1[0, :][i]
    #             a[1, :] = _logTensor_lt1[1, :][i]
    #             result = self.addition(result, a)

    #     if axis == 2:
    #         _logTensor_lt1 = _logTensor_lt1.transpose((0, 2, 1))

    #     return result

    # def broadcast(self, _logTensor_lt1, numberOfCasts, axis=1):

    #     x = _logTensor_lt1
    #     for i in range(numberOfCasts):
    #         _logTensor_lt1 = np.append(_logTensor_lt1, x, axis=axis)

    #     return _logTensor_lt1

class log_multiplication:
    def __init__(self, addObj):
        self.addObj = addObj

    def elem_mult(self, _logTensor_lt1, _logTensor_lt2):
        assert _logTensor_lt1.shape[0] == 2, "First Log-Tensor shape is non-standard"
        assert len(_logTensor_lt1.shape) == 3, "First Log-Tensor shape is non-standard"
        assert _logTensor_lt2.shape[0] == 2, "Second Log-Tensor shape is non-standard"
        assert len(_logTensor_lt2.shape) == 3, "Second Log-Tensor shape is non-standard"
        assert _logTensor_lt1.shape == _logTensor_lt2.shape, "Non-conformable matrices for element wise multiplication"

        result = np.zeros(_logTensor_lt1.shape)
        result[0, :] = (_logTensor_lt1[0, :] == _logTensor_lt2[0, :])
        result[1, :] = _logTensor_lt1[1, :] + _logTensor_lt2[1, :]

        return result

    def scaling_mult(self, _logTensor_lt1, const):
        assert _logTensor_lt1.shape[0] == 2, "Log-Tensor shape is non-standard"
        assert len(_logTensor_lt1.shape) == 3, "Log-Tensor shape is non-standard"

        result = np.zeros(_logTensor_lt1.shape)

        if const > 0:
            val = np.log2(const)
            result[0, :] = _logTensor_lt1[0, :]
        else:
            val = np.log2(-const)
            result[0, :] = 1 - _logTensor_lt1[0, :]

        result[1, :] = _logTensor_lt1[1, :] + val
        return result

    def mattr_mult(self, _logTensor_lt1, _logTensor_lt2):
        assert _logTensor_lt1.shape[0] == 2, "First Log-Tensor shape is non-standard"
        assert len(_logTensor_lt1.shape) == 3, "First Log-Tensor shape is non-standard"
        assert _logTensor_lt2.shape[0] == 2, "Second Log-Tensor shape is non-standard"
        assert len(_logTensor_lt2.shape) == 3, "Second Log-Tensor shape is non-standard"
        assert _logTensor_lt1.shape[2] == _logTensor_lt2.shape[1], "Non-conformable matrices for multiplication"

        return native_matr_mult_wrapper.fast_multiplier(_logTensor_lt1, _logTensor_lt2)

def softmax(inp):

    # inp_shape = inp.shape
    max_vals = np.max(inp, axis=1)
    max_vals.shape = max_vals.size, 1
    u = np.exp(inp - max_vals)
    v = np.sum(u, axis=1)
    v.shape = v.size, 1
    u = u / v
    return u

def imposter_grad_init(_logTensor_lt1, _logTensor_lt2, learningKit):
    assert _logTensor_lt1.shape[0] == 2, "Log Tensor shape is non-standard"
    assert _logTensor_lt2.shape[0] == 2, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt1.shape) == 3, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt2.shape) == 3, "Log Tensor shape is non-standard"

    _, batchsize, num_class = _logTensor_lt1.shape
    m1 = logTensorToMatrix(_logTensor_lt1)
    m2 = softmax(m1)
    my = logTensorToMatrix(_logTensor_lt2)
    # t0 = matrixToLogTensor(m2)
    # m3 = m2 - my
    # ret = learningKit.addObj.subtraction(t0, _logTensor_lt2)
    # ret = matrixToLogTensor(m3)
    m3 = (m2 - my) / batchsize
    # t1 = learningKit.scaling_mult(ret, 1.0 / batchsize)
    # return t1
    return matrixToLogTensor(m3)



def grad_init(_logTensor_lt1, _logTensor_lt2, learningKit):
    assert _logTensor_lt1.shape[0] == 2, "Log Tensor shape is non-standard"
    assert _logTensor_lt2.shape[0] == 2, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt1.shape) == 3, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt2.shape) == 3, "Log Tensor shape is non-standard"

    _, batchsize, num_class = _logTensor_lt1.shape
    retval = np.zeros((2, batchsize, num_class))
    exp_lt1 = np.ones((2, batchsize, num_class))
    exp_lt1[1] = logTensorToMatrix(_logTensor_lt1) * np.log2(np.e)
    denom = np.ones((2, batchsize, 1))
    denom[1] = -np.inf

    denomT = np.zeros(denom.transpose((0, 2, 1)).shape)
    denomT[0] = denom[0].T
    denomT[1] = denom[1].T

    # _lt1T = exp_lt1.transpose((2, 0, 1))
    _lt1T = np.zeros(exp_lt1.transpose((0, 2, 1)).shape)
    _lt1T[0] = exp_lt1[0].T
    _lt1T[1] = exp_lt1[1].T

    u = np.zeros((2, 1, batchsize))
    for i in range(num_class):
        # u = _lt1T[i].reshape((2, 1, batchsize))
        u[0] = _lt1T[0][i]
        u[1] = _lt1T[1][i]
        denomT = learningKit.addObj.addition(u, denomT)

    denom[0] = denomT[0].T
    denom[1] = denomT[1].T
    denom[1] = - denom[1]                   # Combine
    u = np.zeros((2, 1, num_class))
    denomL = logTensorToMatrix(denom)

    for i in range(batchsize):
        u[0][0] = exp_lt1[0][i]
        u[1][0] = exp_lt1[1][i]
        v = learningKit.scaling_mult(u, denomL[i])
        retval[0][i] = v[0][0]
        retval[1][i] = v[1][0]

    ret = learningKit.addObj.subtraction(retval, _logTensor_lt2)

    return learningKit.scaling_mult(ret, 1.0 / batchsize)

def main(main_params):

    is_training = bool(main_params['is_training'])
    leaking_coeff = float(main_params['leaking_coeff'])
    batchsize = int(main_params['minibatch_size'])
    lr = float(main_params['learning_rate'])
    num_epoch = int(main_params['num_epoch'])
    table_size = int(main_params['table_size'])                 # New
    granularity = int(main_params['granularity'])               # New
    appr_add = approximate_addition(table_size, granularity)
    learningKit = log_multiplication(appr_add)
    native_matr_mult_wrapper.init_error_obj(appr_add.N, appr_add.granularity, appr_add.delP, appr_add.delM)
    ones_log = np.array([np.ones((batchsize, 1)), np.zeros((batchsize, 1))])        # New
    print("Number of entries in table: ", table_size * granularity)
        
    if is_training:
        # load fashion mnist data and split into train and test sets
        # one-hot encoded target column

        file = np.load('./../../datasets/fashion_mnist.npz', 'r') # dataset
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

        x_train_log = matrixToLogTensor(x_train)                        # New
        x_test_log = matrixToLogTensor(x_test)                          # New
        x_val_log = matrixToLogTensor(x_val)                            # New
        y_train_log = matrixToLogTensor(y_train)                        # New
        y_test_log = matrixToLogTensor(y_test)                          # New
        y_val_log = matrixToLogTensor(y_val)                            # New

        # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        W1 = np.random.normal(0, 0.1, (785, 100))
        W2 = np.random.normal(0, 0.1, (101, 10))
        
        W1_log = matrixToLogTensor(W1)                                  # New
        W2_log = matrixToLogTensor(W2)                                  # New

        delta_W1_log = np.zeros(W1_log.shape)                           # New
        delta_W2_log = np.zeros(W2_log.shape)                           # New

        performance = {}
        # performance['loss_train'] = np.zeros(num_epoch)               # Useless
        performance['acc_train'] = np.zeros(num_epoch)
        performance['acc_val'] = np.zeros(num_epoch)

        accuracy = 0.0

        for epoch in range(num_epoch):
            print('At Epoch %d:' % (1 + epoch))
            # loss = 0.0
            for mbatch in range(int(split / batchsize)):

                start = mbatch * batchsize
                
                x = [[], []]                                            # New
                y = [[], []]                                            # New

                x[0] = x_train_log[0][start:(start + batchsize)]            # New
                x[1] = x_train_log[1][start:(start + batchsize)]            # New
                y[0] = y_train_log[0][start:(start + batchsize)]            # New
                y[1] = y_train_log[1][start:(start + batchsize)]            # New
                x = np.array(x)                                         # New
                y = np.array(y)                                         # New

                t1 = log_hstack(ones_log, x)                            # New

                s1 = learningKit.mattr_mult(t1, W1_log)                 # New
                ###################################################
                mask = (s1[0] < 0.5) * leaking_coeff                    # New
                ###################################################
                a1 = [[], []]                                           # New
                a1[0] = s1[0]                                           # New
                a1[1] = s1[1] + mask                                    # New
                a1 = np.array(a1)                                       # New
                t2 = log_hstack(ones_log, a1)                           # New
                s2 = learningKit.mattr_mult(t2, W2_log)                 # New
                # a2 = softmax(s2)                                      # Useless

                # cat_cross_ent = np.log(a2) * y                        # Useless
                # cat_cross_ent[np.isnan(cat_cross_ent)] = 0            # Useless
                # loss -= np.sum(cat_cross_ent)                         # Useless

                # grad_s2 = (a2 - y) / batchsize
                grad_s2 = imposter_grad_init(s2, y, learningKit)                 # New
                ###################################################
                t3 = log_hstack(ones_log, a1)                           # New
                t4 = np.zeros(t3.transpose((0, 2, 1)).shape)            # New
                t4[0] = t3[0].T                                         # New
                t4[1] = t3[1].T                                         # New
                delta_W2_log = learningKit.mattr_mult(t4, grad_s2)      # New
                ###################################################
                t5 = np.zeros(np.array(W2_log.shape) - [0, 1, 0])       # New
                t5[0] = W2_log[0][1:]                                   # New
                t5[1] = W2_log[1][1:]                                   # New
                t6 = np.zeros(t5.transpose((0, 2, 1)).shape)            # New
                t6[0] = t5[0].T                                         # New
                t6[1] = t5[1].T                                         # New
                grad_a1 = learningKit.mattr_mult(grad_s2, t6)           # New
                grad_s1 = np.zeros(grad_a1.shape)                       # New
                grad_s1[0] = grad_a1[0]                                 # New
                grad_s1[1] = grad_a1[1] + mask                          # New
                ################################################### 
                t7 = log_hstack(ones_log, x)                            # New
                t8 = np.zeros(t7.transpose((0, 2, 1)).shape)            # New
                t8[0] = t7[0].T                                         # New
                t8[1] = t7[1].T                                         # New
                delta_W1_log = learningKit.mattr_mult(t8, grad_s1)      # New
                ###################################################
                # grad_x =

                # if not(epoch % 4) and (epoch > 0):
                    # lr /= 4

                t9 = learningKit.scaling_mult(delta_W2_log, lr)         # New
                W2_log = learningKit.addObj.subtraction(W2_log, t9)     # New
                t0 = learningKit.scaling_mult(delta_W1_log, lr)         # New
                W1_log = learningKit.addObj.subtraction(W1_log, t0)     # New

            # loss /= split                                             # Useless
            # performance['loss_train'][epoch] = loss                   # Useless
            # print('Loss at epoch %d: %f' %((1 + epoch), loss))        # Useless
            correct_count = 0
            for mbatch in range(int(split / batchsize)):

                start = mbatch * batchsize
                
                x = [[], []]                                            # New
                y = [[], []]                                            # New

                x[0] = x_train_log[0][start:(start + batchsize)]            # New
                x[1] = x_train_log[1][start:(start + batchsize)]            # New
                y[0] = y_train_log[0][start:(start + batchsize)]            # New
                y[1] = y_train_log[1][start:(start + batchsize)]            # New
                x = np.array(x)                                         # New
                y = np.array(y)                                         # New

                t1 = log_hstack(ones_log, x)                            # New

                s1 = learningKit.mattr_mult(t1, W1_log)                 # New
                ###################################################
                mask = (s1[0] < 0.5) * leaking_coeff                    # New
                ###################################################
                a1 = [[], []]                                           # New
                a1[0] = s1[0]                                           # New
                a1[1] = s1[1] + mask                                    # New
                a1 = np.array(a1)                                       # New
                t2 = log_hstack(ones_log, a1)                           # New
                s2 = learningKit.mattr_mult(t2, W2_log)                 # New

                correct_count += np.sum(np.argmax(y[1], axis=1) == np.argmax(logTensorToMatrix(s2), axis=1))    # New

            accuracy = correct_count / split
            performance['acc_train'][epoch] = 100 * accuracy
            print("Train-set accuracy at epoch %d: %f" % ((1 + epoch), performance['acc_train'][epoch]))

            correct_count = 0
            for mbatch in range(int(x_val.shape[0] / batchsize)):

                start = mbatch * batchsize
                
                x = [[], []]                                            # New
                y = [[], []]                                            # New

                x[0] = x_val_log[0][start:(start + batchsize)]          # New
                x[1] = x_val_log[1][start:(start + batchsize)]          # New
                y[0] = y_val_log[0][start:(start + batchsize)]          # New
                y[1] = y_val_log[1][start:(start + batchsize)]          # New
                x = np.array(x)                                         # New
                y = np.array(y)                                         # New

                t1 = log_hstack(ones_log, x)                            # New

                s1 = learningKit.mattr_mult(t1, W1_log)                 # New
                ###################################################
                mask = (s1[0] < 0.5) * leaking_coeff                    # New
                ###################################################
                a1 = [[], []]                                           # New
                a1[0] = s1[0]                                           # New
                a1[1] = s1[1] + mask                                    # New
                a1 = np.array(a1)                                       # New
                t2 = log_hstack(ones_log, a1)                           # New
                s2 = learningKit.mattr_mult(t2, W2_log)                 # New

                correct_count += np.sum(np.argmax(y[1], axis=1) == np.argmax(logTensorToMatrix(s2), axis=1))    # New

            accuracy = correct_count / x_val.shape[0]
            performance['acc_val'][epoch] = 100 * accuracy
            print("Val-set accuracy at epoch %d: %f\n" % ((1 + epoch), performance['acc_val'][epoch]))

        correct_count = 0
        for mbatch in range(int(x_test.shape[0] / batchsize)):

            start = mbatch * batchsize
                
            x = [[], []]                                            # New
            y = [[], []]                                            # New

            x[0] = x_test_log[0][start:(start + batchsize)]         # New
            x[1] = x_test_log[1][start:(start + batchsize)]         # New
            y[0] = y_test_log[0][start:(start + batchsize)]         # New
            y[1] = y_test_log[1][start:(start + batchsize)]         # New
            x = np.array(x)                                         # New
            y = np.array(y)                                         # New

            t1 = log_hstack(ones_log, x)                            # New

            s1 = learningKit.mattr_mult(t1, W1_log)                 # New
            ###################################################
            mask = (s1[0] < 0.5) * leaking_coeff                    # New
            ###################################################
            a1 = [[], []]                                           # New
            a1[0] = s1[0]                                           # New
            a1[1] = s1[1] + mask                                    # New
            a1 = np.array(a1)                                       # New
            t2 = log_hstack(ones_log, a1)                           # New
            s2 = learningKit.mattr_mult(t2, W2_log)

            correct_count += np.sum(np.argmax(y[1], axis=1) == np.argmax(logTensorToMatrix(s2), axis=1))    # New

        accuracy = 100.0 * (correct_count / x_test.shape[0])
        print('Test-set performance: %f' % accuracy)

        np.savez_compressed('./log_model_fashion_MNIST.npz', W1_log=W1_log, W2_log=W2_log, acc_train=performance['acc_train'], \
            acc_val=performance['acc_val'])

    else:

        file = np.load('./log_model_fashion_MNIST.npz', 'r')
        W1_log = file['W1_log']                                         # New
        W2_log = file['W2_log']                                         # New
        performance = {}
        # performance['loss_train'] = file['loss_train']                # Useless
        performance['acc_train'] = file['acc_train']
        performance['acc_val'] = file['acc_val']
        file.close()

        file = np.load('./../../datasets/fashion_mnist.npz', 'r') # dataset
        x_test = file['test_data']
        y_test = file['test_labels']
        x_test, y_test = shuffle(x_test, y_test)
        file.close()

        x_test_log = matrixToLogTensor(x_test)                          # New
        y_test_log = matrixToLogTensor(y_test)                          # New

        correct_count = 0
        for mbatch in range(int(x_test.shape[0] / batchsize)):
            
            start = mbatch * batchsize
                
            x = [[], []]                                            # New
            y = [[], []]                                            # New

            x[0] = x_test_log[0][start:(start + batchsize)]         # New
            x[1] = x_test_log[1][start:(start + batchsize)]         # New
            y[0] = y_test_log[0][start:(start + batchsize)]         # New
            y[1] = y_test_log[1][start:(start + batchsize)]         # New
            x = np.array(x)                                         # New
            y = np.array(y)                                         # New

            t1 = log_hstack(ones_log, x)                            # New

            s1 = learningKit.mattr_mult(t1, W1_log)                 # New
            ###################################################
            mask = (s1[0] < 0.5) * leaking_coeff                    # New
            ###################################################
            a1 = [[], []]                                           # New
            a1[0] = s1[0]                                           # New
            a1[1] = s1[1] + mask                                    # New
            a1 = np.array(a1)                                       # New
            t2 = log_hstack(ones_log, a1)                           # New
            s2 = learningKit.mattr_mult(t2, W2_log)

            correct_count += np.sum(np.argmax(y[1], axis=1) == np.argmax(logTensorToMatrix(s2), axis=1))    # New

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
        x = range(1, 1 + performance['acc_train'].size)                         # New
        ax.plot(x, performance['acc_train'], 'r')
        ax.plot(x, performance['acc_val'], 'g')
        ax.set_xlabel('Number of Epochs')
        ax.set_ylabel('Accuracy')
        ax.set_title('Test-set Accuracy at %.2f%%' % accuracy)
        plt.suptitle('Validation and Training Accuracies\nTable Size %d' % (table_size * granularity), fontsize=14)
        ax.legend(['train', 'validation'])
        plt.grid(which='both', axis='both', linestyle='-.')

        plt.savefig('accuracy.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', default = False)
    parser.add_argument('--split', default = 50000)
    parser.add_argument('--learning_rate', default = 0.01)
    parser.add_argument('--minibatch_size', default = 5)
    parser.add_argument('--num_epoch', default = 20)
    parser.add_argument('--leaking_coeff', default = -7)
    parser.add_argument('--table_size', default=16)
    parser.add_argument('--granularity', default=1)
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)
