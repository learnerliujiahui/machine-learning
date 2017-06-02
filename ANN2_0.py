#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

__anthor__ = 'Liu jiahui'
# data = 3.5-3.9 2016
# objective: a vector with 10-dimentions , the category number is 5
#        *   *   *   *   *   *   *   *   *   *   ---input layer: 10   weights: 10*7 -->7 * 7
#         \ / .......full connected
#          *    *    *    *    *    *    *       ___hiden layer: 7    weights: 7*5 -->matrix L_1
#                      sigmoid
#              *     *    *    *    *            ---output layer:5     -->matrix L_2
#                      softmax
# activate_function:sigmoid
# loss_function: softmax_loss  = -log(softmax[label])
# minibatch = 1 -->SGD
# backpropagetion: softmax --> sigmoid
# learning_rante = step_size
class layer(object):
    # in the layer:
    # Args:
    #       input: should be a row vector [1 * m]
    #       weights: then matrix, each coloumn is a classifier for certain category [m * n]
    #       bias:  should be a row vector in same shape with input [ 1 * n]
    def __init__(self, weights, bias, input):
        # input is the matrix which input the layer
        self.weights = weights
        self.bias = bias
        self.input = input
    # define the sum process:
    def sum(self):
        # Args:
        #       calculate the linear sum of the input vector product weights ,plus bias
        #       linear_output is [1 * n]
        if self.input.shape[1] == self.weights.shape[0]:
            linear_output = np.add(np.dot(self.input, self.weights), self.bias)
            return linear_output
        else:
            print('the input matrix and the weights could not multiply!!!')

def Sigmoid(linear_output):
    # Args:
    #       linear_output: matrix is the linear combination of the weishts * input + bias
    #       linear_output = self.sum()
    #       line: is the (*,1) shape matrix, documenting the sum of each line
    #       non_linear_output:  give the final output of the softmax function
    non_linear_output = 1.0 / (1.0 + np.exp(-1 * linear_output))
    return non_linear_output

def SoftMax(exp_input):
        # nomorlization of the output
        # Args:
        #       exp_input: the exp-lized vector
        #       line_sum: the sum of the exp_lized vector
        #       non_linear_output: the result of softmax
    line_sum = np.sum(exp_input, axis=1)
    non_linear_output = np.zeros((exp_input.shape[0], exp_input.shape[1]))
    for i in range(0, exp_input.shape[0]):
        for j in range(0, exp_input.shape[1]):
            non_linear_output[i, j] = exp_input[i, j] / line_sum[i]
    return non_linear_output

def CrossEntropyDerivative(input, softMaxDot,softMax, label, arg = 'weights'):
    groundTruth = np.zeros((M, 1))
    groundTruth[label] = 1

    if arg == 'weights':
        dotCrossEntropyWeight = np.zeros((L, 1))
        for i in range(0, L):
            for k in range(0, M):
                dotCrossEntropyWeight[i] += input[i] * softMaxDot[k] * (softMax[k]-groundTruth[k])/(softMax[k] *
                                                                                                    (1.0 - softMax[k]))
        return dotCrossEntropyWeight

    if arg == 'bias':
        dotCrossEntropyBias = 0.0
        for k in range(0, M):
            dotCrossEntropyBias += softMaxDot[k] * (softMax[k] - groundTruth[k]) / (softMax[k] * (1.0 - softMax[k]))
        return dotCrossEntropyBias

    if arg == 'x':
        # here, the input vector is one column of the weight matric
        dotCrossEntropyInput = 0.0
        for k in range(0, M):
            dotCrossEntropyInput += softMaxDot[k] * (softMax[k] - groundTruth[k]) / (softMax[k] * (1.0 - softMax[k]))
        return dotCrossEntropyInput

# def sofmax_loss(input, label):
#     # Args:
#     #       input: the softmax vector
#     #       label: the position of the correct category
#     #       loss: the loss value of the correct category
#     loss = -1 * np.log(input[label])
#     return loss

def CrossEntropy(result, label, dimention):
    # Args:
    #       input: the anticipated value of the model
    #       label: ground truth
    #       dimention: the channel of the output result
    groundTruth = np.zeros((dimention, 1))
    groundTruth[label] = 1.0
    loss = 0
    for i in range(0, dimention):
        loss += -( groundTruth[i] * np.log(result[i]) + (1-groundTruth[i]) * np.log(1-result[i]))
    return loss

# back propagation
def SigmoidDerivative( input, output, column,dimention, arg = 'weights'):
        # Args:
        #   input:
        #   if find the derivative of the weights and bias: input--> ventor
        #   if ind the derivative of the x
        #   output: the final result of the forward calculation
        #   dot: used for documenting the gradient
        #   non_linear_output = 1 / (1 + np.exp(-1*output))
    if arg == 'weights':
        dotWeight = np.zeros((dimention, 1))
        # Args:
        #   input：the vector 1 * 10 --> reshape:[10 * 1]
        #   dimention is 10
        #   output is the linear_output of the hidden layer -->is a solid number
        for i in range(0, dimention):
            dotWeight[i] = input[i] * Sigmoid(output[column]) * (1.0 - Sigmoid(output[column]))
        return dotWeight
    if arg == 'bias':
        dotSigmoidBias = Sigmoid(output[column]) * (1.0 - (Sigmoid(output[column])))
        return dotSigmoidBias

def SoftMaxDerivaive(weight, softMax, dimention, column, arg = 'weights'):
        # e.x.:
        # I am going to find the dot of the bias[0] (the first term of the bias,
        # if softMax result is:[0.1,0.2,0.3,0.2,0.2] )
        # and the derivative of the second item in the softmax result is different with the first item
        # -------------------there is nothing to do with the label----------------
        # Args:
            # input: the result after softMax process
            # dimention: sum of the columns
            # label: the label of the current vector, in detailed, the label means which dimention should be the Maximum
    if arg == 'x':
        # weight matrix is the weights_2
        # dimention is M here
        dotinput = np.zeros((M, 1))
        for j in range(0, M):
            dotinput[j] = weight[column, j] * softMax[j]
            for k in range(0, M):
                dotinput[j] += weight[column, k] * softMax[k] * softMax[j]
        return dotinput

    elif arg == 'weights':
        # calculate for one column of the weight matrix
        dotWeight = np.zeros((dimention, 1))
        for i in range(0, dimention):
            if column == i:  # the column undated is exactly the loss one
                dotWeight[i] = softMax[i] * (1 - softMax[i])
            if column != i:  # the other columns
                dotWeight[i] = -1 * softMax[i] * softMax[column]
        return dotWeight

    elif arg == 'bias':
        dotBias = np.zeros((M, 1))
        for i in range(0, M):
            if i == column:
                dotBias[i] = softMax[i] * (1 - softMax[i])
            else:
                dotBias[i] = -1 * softMax[i] * softMax[column]
        return dotBias



if __name__=='__main__':

    # generate the initial vectors, with 5 categories (different mean)
    module_1 = abs(np.random.randn(1000, 2))
    module_2 = abs(np.random.randn(1000, 2) + 5)
    vector_1 = np.column_stack((module_2, module_1, module_1, module_1, module_1))
    vector_2 = np.column_stack((module_1, module_2, module_1, module_1, module_1))
    vector_3 = np.column_stack((module_1, module_1, module_2, module_1, module_1))
    vector_4 = np.column_stack((module_1, module_1, module_1, module_2, module_1))
    vector_5 = np.column_stack((module_1, module_1, module_1, module_1, module_2))
    # vector_3 = np.reshape(abs(np.random.randn(10000) + 8), (1000, 10))
    # vector_4 = np.reshape(abs(np.random.randn(10000) + 12), (1000, 10))
    # vector_5 = np.reshape(abs(np.random.randn(10000) + 16), (1000, 10))
    vector = np.row_stack((vector_1, vector_2, vector_3, vector_4, vector_5))
    # generate the labels:
    label_1 = np.full((1000, 1), 1)
    label_2 = np.full((1000, 1), 2)
    label_3 = np.full((1000, 1), 3)
    label_4 = np.full((1000, 1), 4)
    label_5 = np.full((1000, 1), 5)
    label = np.row_stack((label_1, label_2, label_3, label_4, label_5))
    vector_new = np.hstack((vector, label))
    vector_train = np.zeros((5000, 11))
    for i in range(0, 1000):
        for j in range(0, 5):
            vector_train[i*5 + j] = vector_new[i+1000*j]


    # iniailize the weights matrix and bias for the first and second layer of the ANN
    K = 10  # first_layer_neuron_number
    L = 12  # first_layer_neuron_number
    M = 5   # first_layer_neuron_number
    weights_1 = abs(0.001 * np.random.randn(K, L))
    bias_1 = abs(0.001 * np.random.randn(1, L))
    weights_2 = abs(0.001 * np.random.randn(L, M))
    bias_2 = abs(0.001 * np.random.randn(1, M))

    # weights_1 = np.zeros((K, L))  # for the weight matrix, each column is a classifier for a certain category
    # bias_1 = np.zeros((1, L))
    # weights_2 = np.zeros((L, M))
    # bias_2 = np.zeros((1, M))

    # setting the hyper-parameters
    size_step = 0.01
    epoch = 2
    loss_list = []
    for j in range(0, epoch):
        for i in range(0, 5000):
            # initialize trans matrix
            # linear_output_1 = np.zeros((1, L))
            # non_linear_output_1 = np.zeros((1, L))
            # linear_output_2 = np.zeros((1, M))
            # non_linear_output_2 = np.zeros((1, M))
        # initial the hiden layer of the ANN
        # class args:
            # (weights, bias, input)
            hidden_layer = layer(weights_1, bias_1, np.reshape(vector_train[i, 0:10], (1, K)))
            linear_output_1 = hidden_layer.sum()
            non_linear_output_1 = Sigmoid(linear_output_1)
        # forward to the output layer
        # initial the output layer of the ANN
            output_layer = layer(weights_2, bias_2, non_linear_output_1)
            linear_output_2 = output_layer.sum()
            exp_output_2 = np.exp(linear_output_2)
            non_linear_output_2 = SoftMax(exp_output_2)

        # finish the forward process, and calculate the loss score
            loss_scores = CrossEntropy(np.reshape(non_linear_output_2, (M, 1)), int(vector_train[i, 10]-1), 5)
            loss_list.append(float(loss_scores))  # for the matplotlib
            # print('the number is', i + 5000*j ,' aveerage loss is now:', loss_scores)  # test -->find that around 0.2 is correct

        # using SGD processing'-->training process
        #
        # update the parameters for the output layer:
            dotBias2 = np.zeros((M, 1))
            for column in range(0, M):
                # SoftMaxDerivative(weight, softMax, dimension, column, arg='weights'):
                # CrossEntropyDerivative(input, softMaxDot,dimension, softMax, label, arg = 'weights'):
                dotSoftMAxWeight2 = SoftMaxDerivaive(weights_2, np.reshape(non_linear_output_2, (M, 1)),
                                                     M, column)
                dotSoftMaxBias2 = SoftMaxDerivaive(weights_2, np.reshape(non_linear_output_2, (M, 1)),
                                                     M, column, arg='bias')

                dotWeight2 = CrossEntropyDerivative(np.reshape(non_linear_output_1, (L, 1)),
                                                    np.reshape(dotSoftMAxWeight2, (M, 1)),
                                                    np.reshape(non_linear_output_2, (M, 1)),
                                                    int(vector_train[i, 10])-1)
                dotBias2[column] = CrossEntropyDerivative(np.reshape(non_linear_output_1, (L, 1)),
                                                          np.reshape(dotSoftMaxBias2, (M, 1)),
                                                          np.reshape(non_linear_output_2, (M, 1)),
                                                          int(vector_train[i, 10])-1, arg='bias')
                weights_2[:, column] += -1 * np.reshape(dotWeight2, (L,)) * size_step
            bias_2 += -1 * dotBias2.T * size_step

            dotInput = np.zeros((L, 1))
            for column in range(0, L):
                dotSoftMaxInput = SoftMaxDerivaive(weights_2, np.reshape(non_linear_output_2, (M, 1)),
                                                   M, column, arg='x')
                dotInput[column] = CrossEntropyDerivative(np.reshape(non_linear_output_1, (L, 1)),
                                                          np.reshape(dotSoftMaxInput, (M, 1)),
                                                          np.reshape(non_linear_output_2, (M, 1)),
                                                          int(vector_train[i, 10])-1, arg='x')

            dotBias1 = np.zeros((L, 1))
            for column in range(0, L):
                # SigmoidDerivative( input, output,column, dimension, arg = 'weights'):
                dotSigmoidWeight1 = SigmoidDerivative(np.reshape(vector_train[i, 0:10], (10, 1)),
                                                      np.reshape(linear_output_1, (L, 1)), column, K)
                dotWeight1 = dotSigmoidWeight1 * dotInput[column]
                weights_1[:, column] += -1 * np.reshape(dotWeight1, (K,)) * size_step

                dotSigmoidBias1 = SigmoidDerivative(np.reshape(vector_train[i, 0:10], (10, 1)),
                                                            np.reshape(linear_output_1, (L, 1)), column, K,
                                                            arg= 'bias')
                dotBias1[column] = dotSigmoidBias1 * dotInput[column]
            bias_1 += -1 * dotBias1.T * size_step
    # # visualization of the loss scores data
    fig = plt.figure()
    number = [i for i in range(0, 5000*epoch)]
    plt.plot(number, loss_list, color="blue", linewidth=1.5)
    plt.xlabel("order of the input vector")
    plt.ylabel("Loss Scroe")
    plt.title("the changing of the loss scores each time")
    plt.show()
    #
    count_train = 0
    for i in range(0, 5000):
        hidden_layer_train = layer(weights_1, bias_1, np.reshape(vector_train[i, 0:K], (1, K)))
        linear_output_train_1 = hidden_layer_train.sum()
        non_linear_output_test_1 = Sigmoid(linear_output_train_1)

        output_layer_train = layer(weights_2, bias_2, non_linear_output_test_1)
        linear_output_train_2 = output_layer_train.sum()
        if np.argmax(linear_output_train_2) == int(vector_train[i,K]) - 1:
            count_train += 1
    accuracy_train = count_train / 5000.0
    print('the accuracy is :', accuracy_train, 'in the train database')

    # generate the initial vectors
    module_3 = abs(np.random.randn(1000, 2))
    module_4 = abs(np.random.randn(1000, 2) + 5)
    vector_test_1 = np.column_stack((module_4, module_3, module_3, module_3, module_3))
    vector_test_2 = np.column_stack((module_3, module_4, module_3, module_3, module_3))
    vector_test_3 = np.column_stack((module_3, module_3, module_4, module_3, module_3))
    vector_test_4 = np.column_stack((module_3, module_3, module_3, module_4, module_3))
    vector_test_5 = np.column_stack((module_3, module_3, module_3, module_3, module_4))
    vector_test = np.row_stack((vector_test_1, vector_test_2, vector_test_3, vector_test_4, vector_test_5))
    label_test_1 = np.full((1000, 1), 1)
    label_test_2 = np.full((1000, 1), 2)
    label_test_3 = np.full((1000, 1), 3)
    label_test_4 = np.full((1000, 1), 4)
    label_test_5 = np.full((1000, 1), 5)
    label_test = np.row_stack((label_test_1, label_test_2, label_test_3, label_test_4, label_test_5))
    # # the validation of the module
    count_test = 0
    for i in range(0, 5000):
        hidden_layer_test = layer(weights_1, bias_1, np.reshape(vector_test[i], (1, K)))
        linear_output_test_1 = hidden_layer_test.sum()
        non_linear_output_test_1 = Sigmoid(linear_output_test_1)

        output_layer_test = layer(weights_2, bias_2, non_linear_output_test_1)
        linear_output_test_2 = output_layer_test.sum()
        if np.argmax(linear_output_test_2) == label_test[i]-1:
            count_test += 1
    accuracy_test = count_test/5000.0
    print('the accuracy is :', accuracy_test, 'in the test database')