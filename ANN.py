import numpy as np
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
        #       calculate the linear sum of the input weights ,plus bias
        #       linear_output is [1 * n]
        if self.input.shape[1] == self.weights.shape[0]:
            linear_output = np.add(np.dot(self.input, self.weights), self.bias)
            return linear_output
        else:
            print('the input matrix and the weights could not multiply!!!')

def sigmoid(linear_output):
    # Args:
    #       linear_output: matrix is the linear combination of the weishts * input + bias
    #       linear_output = self.sum()
    #       line: is the (*,1) shape matrix, documenting the sum of each line
    #       non_linear_output:  give the final output of the softmax function

    non_linear_output = 1.0 / (1.0 + np.exp(-1 * linear_output))
    return non_linear_output

def softmax(exp_input):
        # Args:
        #       exp_input: the exp-lized vector
        #       line_sum: the sum of the exp_lized vector
        #       non_linear_output: the result of softmax
    line_sum = np.sum(exp_input,axis=1)
    non_linear_output = np.zeros((exp_input.shape[0], exp_input.shape[1]))
    for i in range(0, exp_input.shape[0]):
        for j in range(0, exp_input.shape[1]):
            non_linear_output[i, j] = exp_input[i, j] / line_sum[i]
    return non_linear_output

def sofmax_loss(input,label):
        # Args:
        #       input: the softmax vector
        #       label: the position of the correct category
        #       loss: the loss value of the correct category
    loss = -1 * np.log(input[label])
    return loss

# back propagation
def sigmoid_dot(input,output,dimention,dot_pre,index,arg = 'weights'):
        # Args:
        #   input:
        #   if find the derivative of the weights and bias: input--> ventor
        #   if ind the derivative of the x
        #   output: the final result of the forward calculation
        #   dot: used for documenting the gradient
        #   non_linear_output = 1 / (1 + np.exp(-1*output))
    dot = np.zeros((dimention, 1))
    if arg == 'weights':
        # Args:
        #   input：the vector 1 * 10 --> reshape:[10 * 1]
        #   dimention is 10
        #   output is the linear_output of the hidden layer -->is a solid number
        for i in range(0, dimention):
            dot[i] = -1 * input[i] * sigmoid(output) * (1.0 - sigmoid(output)) * dot_pre[index]
        return dot
    if arg == 'bias':
        dot_number = -1 * (sigmoid(output)) * (1.0 - (sigmoid(output))) * dot_pre[index]
        return dot_number

def softmax_dot(input, output, label, p, dimention, arg = 'weights'):
        # Args:
        # input: the linear output of the output layer, should be a coloumn vector
        # dot: used for documenting the gradient for one inpu
    dot = np.zeros((dimention, 1))
    line_sum = np.sum(output, axis=1)
    if arg =='x':
        # input matrix is the weights_2
        # output : the exp_output matrix
        # dimention is 5
        for i in range(0, dimention):
            colomun_sum = np.sum(input[i] * output, axis=1)
            dot[i] = -1 * (input[i,label] * line_sum[0]-(colomun_sum[0]))/ (line_sum[0])
        #print('this is for the ...')
        return dot
    if arg == 'weights':
        if label == p: # the colomun undated is eaxactly the loss one
            for i in range(0,dimention):
                dot[i] = -1 * input[i] * (1-output[label]/line_sum[0])
            #print('hello,softmax_main_weights')
            return dot
        if label != p: # the other coloumn
            for i in range(0, dimention):
                dot[i] = input[i] * output[p] / line_sum[0]
            #print('hello,softmax_others_weights')
            return dot
    if arg == 'bias':
        if label == p:
            dot_number = -1 * (1-(output[label] / line_sum[0]))
            return dot_number
        if label != p:
            dot_number = output[p]/(line_sum[0])
        #print('hello,softmax_bias')
            return dot_number

# def value_update(values,dot,step_size = 0.01):
#     values += -step_size * dot
#     return values

if __name__ == '__main__':
    # generate the initial vectors, with 5 categories (different mean)
    vector_1 = np.reshape(abs(np.random.randn(10000)), (1000, 10))
    vector_2 = np.reshape(abs(np.random.randn(10000) + 4), (1000, 10))
    vector_3 = np.reshape(abs(np.random.randn(10000) + 8), (1000, 10))
    vector_4 = np.reshape(abs(np.random.randn(10000) + 12), (1000, 10))
    vector_5 = np.reshape(abs(np.random.randn(10000) + 16), (1000, 10))
    vector = np.row_stack((vector_1, vector_2, vector_3, vector_4, vector_5))
    # generate the labels:
    label_1 = np.full((1000, 1), 3)
    label_2 = np.full((1000, 1), 2)
    label_3 = np.full((1000, 1), 1)
    label_4 = np.full((1000, 1), 4)
    label_5 = np.full((1000, 1), 0)
    label = np.row_stack((label_1,label_2,label_3,label_4,label_5))
    # generate the initial vectors
    vector_test_1 = np.random.randint(0, 1, size=(10, 10))
    vector_test_2 = np.random.randint(3, 6, size=(10, 10))
    vector_test_3 = np.random.randint(6, 10, size=(10, 10))
    vector_test_4 = np.random.randint(10, 14, size=(10, 10))
    vector_test_5 = np.random.randint(14, 18, size=(10, 10))
    vector_test = np.row_stack((vector_test_1, vector_test_2, vector_test_3, vector_test_4, vector_test_5))

    # iniailize the weights matrix and bias for the first and second layer of the ANN
    # weights_1 = np.reshape(abs(0.2 * np.random.randn(70)), (10, 7))
    # bias_1 = np.reshape(abs(0.2 * np.random.randn(7)), (1, 7))
    # weights_2 = np.reshape(abs(0.2 * np.random.randn(35)), (7, 5))
    # bias_2 = np.reshape(abs(0.2 * np.random.randn(5)), (1, 5))
    K = 10  #first_layer_neuron_number
    L = 12  # first_layer_neuron_number
    M = 5  # first_layer_neuron_number
    weights_1 = np.zeros((K,L))
    bias_1 = np.zeros((1, L))
    weights_2 = np.zeros((L, M))
    bias_2 = np.zeros((1, M))
    # seting the hyper-parameters
    size_step = 0.03
    epoch = 10
    for j in range(0,epoch):
        count = 0  # as a number to document the correct number
        for i in range(0, 5000):
            # initialize trans matrix
            linear_output_1 = np.zeros((1, L))
            non_linear_output_1 = np.zeros((1, L))
            linear_output_2 = np.zeros((1, M))
            non_linear_output_2 = np.zeros((1, M))
        # class args:
            # (weights, bias, input)
            hidden_layer = layer(weights_1, bias_1, np.reshape(vector[i],(1,K)))
            linear_output_1 = hidden_layer.sum()
            non_linear_output_1 = sigmoid(linear_output_1)
        # forward to the second layer
            output_layer = layer(weights_2, bias_2, non_linear_output_1)
            linear_output_2 = output_layer.sum()
            exp_output_2 = np.exp(linear_output_2)
            non_linear_output_2 = softmax(exp_output_2)
        # finish the forward process, and calculate the loss score
            loss_scores = sofmax_loss(np.reshape(non_linear_output_2,(M,1)),label[i])
            if np.argmax(non_linear_output_2) == label[i]:
                count += 1
            #print('the number is',i+ 5000*j,' aveerage loss is now:',loss_scores) #test -->find that around 0.2 is correct

        # using SGD processing'-->training process
        #
        # update the paremeters for the output layer:
            dot_bias_2 = np.zeros((M, 1))
            for p in range(0, M):
                # softmax_dot(input, output, label, p, dimention, arg = 'weights'):
                dot_weights_2 = softmax_dot(np.reshape(non_linear_output_1, (L, 1)), np.reshape(exp_output_2, (M, 1)), label[i], p, L)
                dot_bias_2[p] = softmax_dot(np.reshape(non_linear_output_1, (L, 1)), np.reshape(exp_output_2, (M, 1)), label[i], p, M, arg = 'bias')
                weights_2[:, p] += -1 * size_step * np.reshape(dot_weights_2, (L,))
            bias_2 += -1 * size_step * dot_bias_2.T

        # update the patameter for the first layer:
            dot_bias_1 = np.zeros((L,1))
            for m in range(0,L):
                dot_weights_2_1 = softmax_dot(weights_2, np.reshape(exp_output_2, (M, 1)), label[i], m, L, arg='x')
                # sigmoid_dot(input,output,dimention,dot_pre,index,arg = 'weights')
                dot_weights_1 = sigmoid_dot(np.reshape(vector[i], (K, 1)), linear_output_1[:,m], K, dot_weights_2_1, m)
                dot_bias_1[m] = sigmoid_dot(np.reshape(vector[i], (K, 1)), linear_output_1[:,m], L, dot_weights_2_1, m, arg = 'bias')
                weights_1[:, m] += -1 * size_step * np.reshape(dot_weights_1, (10,))
            bias_1 += -1 * size_step * dot_bias_1.T
        accuracy = count / 5000.0
        print('this is the ',j+1,'epoch,and accuracy is',accuracy)
