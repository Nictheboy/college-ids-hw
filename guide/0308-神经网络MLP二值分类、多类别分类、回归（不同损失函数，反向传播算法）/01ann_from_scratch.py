import numpy as np
from keras.datasets import mnist
#import datasets.mnist.loader as mnist
import matplotlib.pylab as plt
 
class ANN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
 
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
 
    def initialize_parameters(self):
        np.random.seed(1)
 
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    def forward(self, X):
        store = {}
 
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z
 
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.sigmoid(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
 
        return A, store
 
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
 
    def backward(self, X, Y, store):
 
        derivatives = {}
 
        store["A0"] = X.T
 
        A = store["A" + str(self.L)]
        dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)
 
        dZ = dA * self.sigmoid_derivative(store["Z" + str(self.L)])
        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
 
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)
 
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
 
        return derivatives
 
    def fit_with_init(self, X, Y, learning_rate=0.01, n_iterations=2500):
        np.random.seed(1)
 
        self.n = X.shape[0]
 
        self.layers_size.insert(0, X.shape[1])
 
        self.initialize_parameters()
        for loop in range(n_iterations):
            A, store = self.forward(X)
            cost = np.squeeze(-(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / self.n)
            derivatives = self.backward(X, Y, store)
 
            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]
 
            if loop % 100 == 0:                
                print("cost",cost)
                self.costs.append(cost)

    def fit_more(self, X, Y, learning_rate=0.01, n_iterations=2500):
        
        for loop in range(n_iterations):
            A, store = self.forward(X)
            cost = np.squeeze(-(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / self.n)
            derivatives = self.backward(X, Y, store)
 
            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]
 
            if loop % 100 == 0:                
                print("cost",cost)
                self.costs.append(cost)
                
    def predict(self, X, Y):
        A, cache = self.forward(X)
        n = X.shape[0]
        p = np.zeros((1, n))
 
        for i in range(0, A.shape[1]):
            if A[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
 
        print("Accuracy: " + str(np.sum((p == Y) / n)))
    def predict_one(self,X):
        A, cache = self.forward(X)
        n = X.shape[0]
        p = np.zeros((1, n))
        for i in range(0, A.shape[1]):
            if A[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        
        return p
    
    def plot_cost(self):
        plt.figure(1)
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()
 
 
def get_binary_dataset():
    #train_x_orig, train_y_orig, test_x_orig, test_y_orig = mnist.get_data()
    (train_x_orig, train_y_orig), (test_x_orig, test_y_orig) = mnist.load_data()
 
    index_5 = np.where(train_y_orig == 5)
    index_8 = np.where(train_y_orig == 8)
 
    index = np.concatenate([index_5[0], index_8[0]])
    np.random.seed(1)
    np.random.shuffle(index)
 
    train_y = train_y_orig[index]
    train_x = train_x_orig[index]
 
    train_y[np.where(train_y == 5)] = 0
    train_y[np.where(train_y == 8)] = 1
 
    index_5 = np.where(test_y_orig == 5)
    index_8 = np.where(test_y_orig == 8)
 
    index = np.concatenate([index_5[0], index_8[0]])
    np.random.shuffle(index)
 
    test_y = test_y_orig[index]
    test_x = test_x_orig[index]
 
    test_y[np.where(test_y == 5)] = 0
    test_y[np.where(test_y == 8)] = 1
 
    return train_x, train_y, test_x, test_y
 
def pre_process_data(train_x, test_x):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.
 
    return train_x, test_x
 
 
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_binary_dataset()
 
    train_x, test_x = pre_process_data(train_x, test_x)
    dim0 = train_x.shape[0]
    dim1 = train_x.shape[1]
    dim2 = train_x.shape[2]
    train_x= train_x.reshape(dim0,dim1*dim2)

    dim0 = test_x.shape[0]
    dim1 = test_x.shape[1]
    dim2 = test_x.shape[2]
    test_x = test_x.reshape(dim0,dim1*dim2)

    print("train_x's shape: " + str(train_x.shape))
    print("train_y's shape: " + str(train_y.shape))
    print("train_x",train_x)
    print("train_y",train_y)

    print("test_x's shape: " + str(test_x.shape))
    print("test_y's shape: " + str(test_y.shape))
    
    layers_dims = [196, 1]
 
    ann = ANN(layers_dims)
    dim0 = train_x.shape[0]
    train_x_0 = train_x[0:int(dim0*0.2) ]
    train_y_0 = train_y[0:int(dim0*0.2) ]
    train_x_1 = train_x[int(dim0*0.2): int(dim0*0.4)]
    train_y_1 = train_y[int(dim0*0.2): int(dim0*0.4 )]
    train_x_2 = train_x[int(dim0*0.4): int(dim0*0.6)]
    train_y_2 = train_y[int(dim0*0.4): int(dim0*0.6)]
    train_x_3 = train_x[int(dim0*0.6): int(dim0*0.8)]
    train_y_3 = train_y[int(dim0*0.6): int(dim0*0.8)]
    train_x_4 = train_x[int(dim0*0.8):]
    train_y_4 = train_y[int(dim0*0.8):]
    
    if True:
        ann.fit_with_init(train_x_0, train_y_0, learning_rate=0.1, n_iterations=500)
    
        #ann.fit(train_x_2, train_y_2, learning_rate=0.1, n_iterations=1000)
        #ann.fit(train_x_3, train_y_3, learning_rate=0.1, n_iterations=1000)
        #ann.fit(train_x_4, train_y_4, learning_rate=0.1, n_iterations=1000)
    ann.fit_more(train_x_1, train_y_1, learning_rate=0.1, n_iterations=500)
        
    #ann.fit(train_x, train_y, learning_rate=0.1, n_iterations=500)    
    ann.predict(train_x, train_y)
    ann.predict(test_x, test_y)
    ann.plot_cost()

    from matplotlib import pyplot
    pyplot.figure(2)
    for i in range(9):  
        pyplot.subplot(330 + 1 + i)
        one_test = test_x[i]
        one_test = one_test .reshape(28,28)
        pyplot.imshow(one_test , cmap=pyplot.get_cmap('gray'))
        pyplot.show()

    nine_test = test_x[:9]
    nine_y = ann.predict_one(nine_test)
    print("predicted nine_y",nine_y)
    print("old nine_y",test_y[:9])
    