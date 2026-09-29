"""
This module defines important functions that mimic matrix operations, like element wise addition, element wise multiplication, constant-matrix multiplication and matrix-matrix multiplication in log domain. This module also provides functions that convert matrices to log domain tensors and vice versa. The neural net matrices become rank 3 tensors in log domain. The shape of the tensor is the same as the shape of two stacked input matrices.

Data structure used - Rank 3 Tensors. The shape of this cube is related to the shape of the original matrix. The height of the cube is 2 as now the cube is just two stacked matrices, the lower stack giving the sign information while the upper stack storing the actual logarithm values in base 2 log. Sign value 1 denotes positive and 0 denotes non-positive. All logarithms are base 2.
"""

import numpy as np
import native_matr_mult_wrapper

def matrixToLogTensor(mattr_a):                             # Bug Free
    assert len(mattr_a.shape) == 2, "Input Matrix shape is non-standard"

    _logTensor_lt1 = np.zeros(mattr_a.size * 2)
    _logTensor_lt1.shape = 2, mattr_a.shape[0], mattr_a.shape[1]
    _logTensor_lt1[0, :] = mattr_a > 0
    _logTensor_lt1[1, :] = np.log2(mattr_a * ((2 * _logTensor_lt1[0, :]) - 1))

    return _logTensor_lt1

def logTensorToMatrix(_logTensor_lt1):                             # Bug Free
    assert _logTensor_lt1.shape[0] == 2, "Log Tensor shape is non-standard"
    assert len(_logTensor_lt1.shape) == 3, "Log Tensor shape is non-standard"

    mattr_a = np.zeros(_logTensor_lt1[0, :].shape)
    mattr_a = (2 ** _logTensor_lt1[1, :]) * ((2 * _logTensor_lt1[0, :]) - 1)

    return mattr_a

def logleaky(_logTensor_lt1, leaking_coefficient):
    assert len(_logTensor_lt1.shape) == 3, "Log Tensor shape is non-standard"
    assert _logTensor_lt1.shape[0] == 2, "Log Tensor shape is non-standard"

    result = np.zeros(_logTensor_lt1.shape)
    result[1, :][_logTensor_lt1[0, :] == 0] = leaking_coefficient
    result[0, :][_logTensor_lt1[0, :] == 1] = 1
    result[1, :][_logTensor_lt1[0, :] == 1] = 0
    # result[1, :] = 0

    return result

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

### Modules ###

class log_linear_layer:
    
    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """

    def __init__(self, input_D, output_D, minB_size, learningKit):
        self.params = dict()
        self.params['W'] = matrixToLogTensor(np.random.normal(0, 0.10, (1 + input_D, output_D)))	# should be 0.0025
        # self.params['b'] = matrixToLogTensor(np.random.normal(0, 0.10, (1, output_D)))	# should be 0.0025
        # self.params['W'] = matrixToLogTensor(np.ones((input_D, output_D)))

        self.gradient = dict()
        self.gradient['W'] = matrixToLogTensor(np.zeros((1 + input_D, output_D)))
        # self.gradient['b'] = matrixToLogTensor(np.zeros((1, output_D)))
        self.learningKit = learningKit
        self.input_D = input_D
        self.output_D = output_D
        self.N = minB_size
        self.log_concat_x = None

    def forward(self, X):

        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (i.e., X[i], where i = 1,...,N).
                The mini-batch size is N.

            Operation:
            - You are going to generate a N-by-output_D numpy array named forward_output.
            - For each row x of X (say X[i]), perform X[i] self.params['W'] + self.params['b'], and store the output in forward_output[i].
            - Please use np.XX to call a numpy function XX.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """

        ################################################################
        # The linear forward pass. Store the result in forward_output  #
        ################################################################

        # forward_output = np.matmul(X, self.params['W']) + self.params['b']
        self.log_concat_x = np.zeros((2, self.N, 1 + self.input_D))
        self.log_concat_x[0] = np.hstack((np.ones((self.N, 1)), X[0]))
        self.log_concat_x[1] = np.hstack((np.zeros((self.N, 1)), X[1]))
        forward_output = self.learningKit.mattr_mult(self.log_concat_x, self.params['W'])
        return forward_output

    def backward(self, X, grad):

        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'], and X.
            - You are going to generate a N-by-input_D numpy array named backward_output.
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. X in backward_output.
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'] in self.gradient['W'].
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['b'] in self.gradient['b'].
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. X[i].
        """

        ##########################################################################################################################
        # The backward pass (computes the following three terms)                                                                 #
        # self.gradient['W'] = ? (input_D-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['W'])  #
        # self.gradient['b'] = ? (1-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['b'])        #
        # backward_output = ? (N-by-input_D numpy array, the gradient of the mini-batch loss w.r.t. X)                           #
        # only return backward_output, but need to compute self.gradient['W'] and self.gradient['b']                             #
        ##########################################################################################################################

        # xt = X.transpose((0, 2, 1))
        xt = matrixToLogTensor(logTensorToMatrix(self.log_concat_x).T)
        # wt = self.params['W'].transpose((0, 2, 1))
        wt = matrixToLogTensor(logTensorToMatrix(self.params['W'])[1:].T)
        self.gradient['W'] = self.learningKit.mattr_mult(xt, grad)
        backward_output = self.learningKit.mattr_mult(grad, wt)
        return backward_output


class log_relu:

    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self, learningKit, leaking_coefficient):
        self.mask = None
        self.learningKit = learningKit
        self.leaking_coefficient = leaking_coefficient

    def forward(self, X):

        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Operation:
            - You are to generate a numpy array named forward_output of the same shape of X.
            - For each element x of X, perform max{0, x}, and store it in the corresponding element of forward_output.
            - Please use np.XX to call a numpy function XX if necessary.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You can use self.mask to store what you may need (except X) for the use in the backward pass.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        #################################################################
        # The relu forward pass. Stores the result in forward_output    #
        #################################################################

        # forward_output = X * (X > 0)
        forward_output = self.learningKit.elem_mult(X, logleaky(X, self.leaking_coefficient))
        return forward_output

    def backward(self, X, grad):

        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Operation:
            - You are to generate a numpy array named backward_output of the same shape of X.
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. X, and store it in backward_output.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You can use self.mask.
            - PLEASE follow the Heaviside step function defined in CSCI567_HW2.pdf

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in  X.
        """

        ##########################################################################################################################
        # The backward pass (computes the following term)                                                                        #
        # backward_output = ? (A numpy array of the shape of X, the gradient of the mini-batch loss w.r.t. X)                    #
        # Hint: The Heaviside step function                                                                                      #
        ##########################################################################################################################

        # backward_output = grad * (X > 0)
        backward_output = self.learningKit.elem_mult(grad, logleaky(X, self.leaking_coefficient))
        return backward_output

### Loss functions ###

class log_softmax_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X1, Y):
        # print("Debug ====== \nX1 inp shape: ", X1.shape)
        X = logTensorToMatrix(X1)
        # if np.sum(np.isnan(X) + (X == np.inf) + (X == -np.inf)) > 0:
            # print("Instability detected in core\n")
            # exit(0)
        # print("Debug ====== \nX inp shape: ", X.shape, Y.size)        
        # self.expand_Y = np.zeros(X.shape)
        # self.expand_Y[range(Y.size), Y.astype(int)] = 1.0
        self.expand_Y = np.zeros(X.shape).reshape(-1)        
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0        
        self.expand_Y = self.expand_Y.reshape(X.shape)      

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit
        # self.prob[self.prob == 0] = 1e-323
        if np.sum(self.prob == 0):                        #N.B. - if things don't work out rendezvous here
            print("Instability detected because pred probs are zero !!\npred prob mattr\n", self.prob, "\n\n", X)
            exit(0)

        # if np.sum(np.isnan(self.prob)) or np.sum(self.prob == -np.inf) or np.sum(self.prob == np.inf):
            # print(self.prob)

        # print("Debug ====== \nself.prob:\n", np.log(self.prob))
        # if np.isnan(np.sum(np.log(self.prob)  * self.expand_Y)):
            # print("Shit\n")
            # exit(0)

        # t = np.log(self.prob)  * self.expand_Y
        # t[np.isnan(t)] = 0.0
        # t[t == -np.inf] = -743.7469247408213
        # forward_output = -np.sum(t) / X.shape[0]     
        forward_output = -np.sum(np.log(self.prob)  * self.expand_Y) / X.shape[0]     
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[1]
        return matrixToLogTensor(backward_output)

class log_lin_loss_function:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.prob = None

    def forward(self, X1, Y):
        X = logTensorToMatrix(X1)

        self.expand_Y = np.zeros(X.shape).reshape(-1)        
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0        
        self.expand_Y = self.expand_Y.reshape(X.shape)

        t = np.min(X, axis=1)
        t.shape = t.shape[0], 1
        self.calib_logit = X - t + 1
        u = np.sum(self.calib_logit, axis=1)
        u.shape = u.shape[0], 1
        self.prob = self.calib_logit/u

        forward_output = -np.sum(np.log(self.prob)  * self.expand_Y) / X.shape[0]     
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[1]
        return matrixToLogTensor(backward_output)

