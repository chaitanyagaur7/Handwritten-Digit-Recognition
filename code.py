import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('data/train.csv')



data = np.array(data)
m, n = data.shape

np.random.shuffle(data) 


data_dev = data[0:1000].T
# Now after transposing the matrix, each column is an example (digit from 0 to 9)
# and we have 784 rows (28px*28px) gives us 784px must be stored for each example


Y_dev = data_dev[0] # The output (the actual digit) is in the first column of the training set


X_dev = data_dev[1:n] #we skipped the 0th row because it is the output (the labels)
X_dev = X_dev / 255.



data_train = data[1000:m].T  # from 1000 to 42000
Y_train = data_train[0] # The output (the actual digit) is in the first column of the training set
X_train = data_train[1:n] # we skipped the 0th row because it is the output (the labels)
X_train = X_train / 255.


_,m_train = X_train.shape




def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)
# This is element-wise, which means it will take every element of the matrix Z and compare it with 0. If the element
# element > 0 then return the element. Else, return 0
# And tha's how the ReLU (Rectified Linear Activation function) works.

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
# Z is a matrix of 10xm.  Each column is an ecample.
# sum(np.exp(z)) returns the sum for each column (the sum of the exponent of each element in that column)
# then we divide each element in the column by that sum.

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0
# If the z > 0, then the derivative of the linear function is 1. Else if the z<=0 then the derivative of the constant is 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
   
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y #  returns a list of 10 rows, each column is an example


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # This is the error
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    # Now for the hidden layer:
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
# -------------------------------------------------------------------
def get_predictions(A2):
    
    return np.argmax(A2, 0) # here we decide which digit this example/image represents based on the higher probability which was calculated using the softmax activation function. 

def get_accuracy(predictions, Y):
    # Y is the list of the desired (expected) outputs: e.g. [9 2 1 ... 5 8 1]
    print(f"prediction: {predictions}, Desired Output: {Y}")
    # predictions is the list that our model has outputted: e.g. [9 2 2 ... 6 9 1]
    return np.sum(predictions == Y) / Y.size
    # np.sum(predictions == Y) returns the number of true predictions


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Accuracy: {round(accuracy, 4)}") # prints the accuracy on the training set after every 10 iterations
            print("-----------------------------------------")
    return W1, b1, W2, b2
# -------------------------------------------------------------------



def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500) 

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
test_prediction(4, W1, b1, W2, b2)
test_prediction(5, W1, b1, W2, b2)
test_prediction(6, W1, b1, W2, b2)
test_prediction(7, W1, b1, W2, b2)
test_prediction(8, W1, b1, W2, b2)
test_prediction(101, W1, b1, W2, b2)
test_prediction(1060, W1, b1, W2, b2)
test_prediction(1065, W1, b1, W2, b2)
test_prediction(106, W1, b1, W2, b2)
test_prediction(1068, W1, b1, W2, b2)
test_prediction(10, W1, b1, W2, b2)



dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Accuracy on the testing set: {accuracy}")
