import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss_mean_square(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

def sigma_prime(x):
    f = sigmoid(x)
    return f * (1-f)

class N:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    #Takes x values
    def feed_forward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
class Network:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = N(weights, bias)
        self.h2 = N(weights, bias)
        self.out = N(weights, bias)

    def feed_forward(self, x):
        out_h1 = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)

        out_out = self.out.feed_forward(np.array([out_h1, out_h2]))

        return out_out

class NewNeural:
    def __init__(self):
        #Weights
        self.weight_hold = []
        for i in range(6):
            self.weight_hold.append(np.random.normal())

        self.bias_hold = []
        for i in range(3):
            self.bias_hold.append(np.random.normal())

    def feedforward(self, x):
        h1 = sigmoid(self.weight_hold[0] * x[0] + 
                     self.weight_hold[1] * x[1] + 
                     self.bias_hold[0])
        
        h2 = sigmoid(self.weight_hold[2] * x[0] +
                     self.weight_hold[3] * x[1] +
                     self.bias_hold[1])
        
        o1 = sigmoid(self.weight_hold[4] * h1 +
                     self.weight_hold[5] * h2 +
                     self.bias_hold[2])
        return o1
    
    def train(self, data, all_y_true):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_true):
                sum_h1 = self.weight_hold[0] * x[0] + self.weight_hold[1] * x[1] + self.bias_hold[0]
                h1 = sigmoid(sum_h1)

                sum_h2 = self.weight_hold[2] * x[0] + self.weight_hold[3] * x[1] + self.bias_hold[1]
                h2 = sigmoid(sum_h2)

                sum_o1 = self.weight_hold[4] * h1 + self.weight_hold[5] * h2 + self.bias_hold[2]
                o1 = sigmoid(sum_o1)

                y_pred = o1

                #Calc Partial Derivatives for backtracking
                d_L_d_ypred = -2 * (y_true - y_pred)

                #Node o
                d_ypred_d_w5 = h1 * sigma_prime(sum_o1)
                d_ypred_d_w6 = h2 * sigma_prime(sum_o1)
                d_ypred_d_b3 = sigma_prime(sum_o1)

                d_ypred_d_h1 = self.weight_hold[4] * sigma_prime(sum_o1)
                d_ypred_d_h2 = self.weight_hold[5] * sigma_prime(sum_o1)

                #Node h1
                d_h1_d_w1 = x[0] * sigma_prime(sum_h1)
                d_h1_d_w2 = x[1] * sigma_prime(sum_h1)
                d_h1_d_b1 = sigma_prime(sum_h1)

                #Node h2
                d_h2_d_w3 = x[0] * sigma_prime(sum_h2)
                d_h2_d_w4 = x[1] * sigma_prime(sum_h2)
                d_h2_d_b2 = sigma_prime(sum_h2)

                #Update weights
                #Node h1
                self.weight_hold[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.weight_hold[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.bias_hold[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                #Node h2
                self.weight_hold[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.weight_hold[3] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.bias_hold[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                #Node o1
                self.weight_hold[4] -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.weight_hold[5] -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.bias_hold[2] -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_pred = np.apply_along_axis(self.feedforward,1,data)
                loss = loss_mean_square(all_y_true, y_pred)
                print(f'Epoch {epoch} loss: {loss}')

#Dataset dummy
data = np.array([
    [-2,-1],
    [25,6],
    [17,4],
    [-15,-6]
])

all_y_true = np.array([
    1,
    0,
    0,
    1
])

weights = np.array([0,1])
bias = 4
n = N(weights=weights, bias=bias)

x = np.array([2,4])
print(n.feed_forward(x))

network = Network()
x = np.array([2,3])
print(network.feed_forward(x))

#Train model
network = NewNeural()
network.train(data, all_y_true)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
