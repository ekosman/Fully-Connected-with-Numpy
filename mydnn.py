import abc
import random
import time
from os import path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn import preprocessing
import pickle, gzip, numpy, urllib.request, json
from keras.utils import to_categorical
import os
import itertools

plt.rcParams['figure.constrained_layout.use'] = True


class Layer:
    def __init__(self, parents):
        self.parents = parents
        self.gradient = None
        self.x = None

    @abc.abstractmethod
    def forward(self, x):
        pass

    def backward(self, grad):
        new_grad = self.calc_grad(grad)
        if self.gradient is None:
            self.gradient = new_grad
        else:
            self.gradient += new_grad

        parent_grad = self.calc_parent_grad(grad)

        if self.parents is None:
            return

        for parent in self.parents:
            parent.backward(parent_grad)

    @abc.abstractmethod
    def calc_parent_grad(self, grad):
        pass

    @abc.abstractmethod
    def calc_grad(self, grad):
        pass

    @abc.abstractmethod
    def update_weights(self, learning_rate):
        pass


class BiasLayer(Layer):
    def __init__(self, parents, shape):
        super().__init__(parents)
        self.bias = np.zeros(shape)
        self.shape = shape

    def calc_parent_grad(self, grad):
        return grad

    def calc_grad(self, grad):
        return np.reshape(np.sum(grad, axis=0, keepdims=True), -1)

    def update_weights(self, learning_rate):
        self.bias -= learning_rate * (preprocessing.normalize(self.gradient.reshape(1, -1), norm='l2').reshape(-1))
        self.gradient = None

    def forward(self, x):
        return x + self.bias


class MatrixLayer(Layer):
    def __init__(self, shape, parents, regularization):
        super().__init__(parents)
        n = shape[1]
        delta = 1 / np.sqrt(n)
        self.weights = np.random.uniform(-delta, delta, shape)
        self.regularization = getattr(self, regularization)

    def forward(self, x):
        self.x = x
        return x @ self.weights

    def l1(self, a):
        self.gradient += a * np.sign(self.weights)

    def l2(self, a):
        self.gradient += (a * 2) * self.weights

    def calc_parent_grad(self, grad):
        return grad @ self.weights.T

    def calc_grad(self, grad):
        return self.x.T @ grad

    def update_weights(self, learning_rate):
        gradient_shape = self.gradient.shape
        self.weights -= learning_rate * np.reshape(preprocessing.normalize(self.gradient.reshape(1,-1), norm='l2'), gradient_shape)
        self.gradient = None


class Identity(Layer):
    def update_weights(self, learning_rate):
        self.gradient = None

    def calc_grad(self, grad):
        return np.zeros(grad.shape)

    def calc_parent_grad(self, grad):
        return grad

    def forward(self, x):
        self.x = x
        return x


class Relu(Layer):
    def update_weights(self, learning_rate):
        self.gradient = None

    def calc_grad(self, grad):
        op = lambda x: 0 if x <= 0 else 1
        np_op = np.vectorize(op)
        return np_op(self.x) * grad

    def calc_parent_grad(self, grad):
        return self.calc_grad(grad)

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)


class Sigmoid(Layer):
    def update_weights(self, learning_rate):
        self.gradient = None

    def calc_grad(self, grad):
        op = lambda x: np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        np_op = np.vectorize(op)
        return np_op(self.x) * grad

    def calc_parent_grad(self, grad):
        return self.calc_grad(grad)

    def forward(self, x):
        self.x = x
        exp_of_x = np.exp(x)
        return exp_of_x / (1 + exp_of_x)


class Softmax(Layer):
    def update_weights(self, learning_rate):
        self.gradient = None

    def forward(self, x):
        self.x = x
        # maxes = np.max(x, axis=-1)
        # maxes = np.repeat([maxes], x.shape[1], axis=0).T
        # exp_of_x = np.exp(x)
        # sums = np.sum(exp_of_x, axis=-1)
        # opossite_sums = 1 / sums
        # division_matrix = np.diag(1 / sums)
        # return division_matrix @ exp_of_x
        s = np.max(x, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(x - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div

    @staticmethod
    def jacobian(x):
        s = np.reshape(x, (-1, 1))
        return np.diagflat(s) - np.dot(s, s.T)

    def calc_grad(self, grad):
        # res = np.zeros(self.x.shape)
        # for i in range(res.shape[0]):
        #     jac = np.array(Softmax.jacobian(self.x[i]))
        #     res[i, :] = np.reshape(grad[i], (1, -1)) @ jac
        #
        # res = np.diag(1 / np.linalg.norm(res, axis=-1)) @ res
        # return res
        return grad

    def calc_parent_grad(self, grad):
        return self.calc_grad(grad)


class mydnn():
    def __init__(self, architecture, loss, weight_decay=0.0):
        self.architecture = architecture
        self.loss = loss
        self.weight_decay = weight_decay
        self.layers = [Identity(None)]
        for layer in architecture:
            m = MatrixLayer((layer["input"], layer["output"]), [self.layers[-1]], layer["regularization"])
            self.layers.append(m)
            b = BiasLayer([m], layer["output"])
            self.layers.append(b)
            a = None
            activation = layer["nonlinear"]
            if activation == "relu":
                a = Relu([b])
            elif activation == "sigmoid":
                a = Sigmoid([b])
            elif activation == "soft-max":
                a = Softmax([b])
            else:
                a = Identity([b])
            self.layers.append(a)

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, learning_rate_decay=1.0, decay_rate=1,
            min_lr=0.0, x_val=None, y_val=None):
        step = 0
        history = []
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            indexes = np.random.permutation(x_train.shape[0])
            permutate_x_train = x_train[indexes]
            permutate_y_train = y_train[indexes]

            batches_X = [permutate_x_train[i * batch_size:(i + 1) * batch_size, :] for i in
                         range(permutate_x_train.shape[0] // batch_size)]
            batches_Y = [permutate_y_train[i * batch_size:(i + 1) * batch_size] for i in
                         range(permutate_y_train.shape[0] // batch_size)]

            epoch_loss, epoch_acc = [], []
            for i in range(len(batches_X)):
                x, y = batches_X[i], batches_Y[i]
                learning_rate = max(learning_rate * learning_rate_decay ** (step / decay_rate), min_lr)
                step += 1
                train_res = self.predict(x)

                grad = None
                if self.loss == 'MSE':
                    grad = train_res - y
                else:
                    # dev_y_hat = 1 / train_res
                    # grad = -dev_y_hat * y
                    grad = train_res - y

                self.layers[-1].backward(grad)

                for layer in self.layers:
                    if isinstance(layer, MatrixLayer):
                        layer.regularization(self.weight_decay)
                    layer.update_weights(learning_rate)

                '''
                Calculate metrics on the current batch
                Metrics on the validation set would be calculated at the end of the epoch
                Note that accuracy is calculate only for 
                '''
                batch_loss, batch_acc = self.evaluate(x, y)  # calculate metrics on the current batch
                epoch_loss.append(batch_loss)
                if self.loss == "cross-entropy":
                    epoch_acc.append(batch_acc)
                # End of training step

            total_epoch_loss = np.mean(epoch_loss)
            if self.loss == "cross-entropy":
                total_epoch_acc = np.mean(epoch_acc)
            else:
                total_epoch_acc = None

            val_loss, val_acc = self.evaluate(x_val, y_val) # calculate metrics on the validation set

            history.append(
                {'step': epoch, 'acc': total_epoch_acc, 'loss': total_epoch_loss, 'val_acc': val_acc,
                 'val_loss': val_loss})

            epoch_time = round(time.time() - start_time, 2)
            print("Epoch {} / {} - {} seconds - loss: {}".format(epoch, epochs, epoch_time, round(total_epoch_loss, 4)), end="")
            if total_epoch_acc is not None:
                print(" - acc: {}".format(round(total_epoch_acc, 4)), end="")
            if val_loss is not None:
                print(" - val_loss: {}".format(round(val_loss, 4)), end="")
            if val_acc is not None:
                print(" - val_acc: {}".format(round(val_acc, 4)), end="")

            print("")

        return history

    @staticmethod
    def subtract_mean(vectors):
        return vectors - np.mean(vectors, axis=0)

    def predict(self, x, batch_size=None):
        res = []
        batch_size = x.shape[0] if batch_size is None else batch_size
        batches = [x[i * batch_size:(i + 1) * batch_size] for i in range(x.shape[0] // batch_size)]
        for batch in batches:
            temp = batch
            for layer in self.layers:
                temp = layer.forward(temp)
            res.append(temp)

        return np.concatenate(res)

    def evaluate(self, x, y, batch_size=None):
        '''
        :param x: The set the evaluate
        :param y: Labels of x
        :param batch_size: Batch size
        :return: Tuple (loss, acc)
        '''
        if x is None or y is None:
            return None, None
        res = self.predict(x, batch_size)
        if self.loss == "MSE":
            return mydnn.calc_mse(y, res), None
        return mydnn.calc_cross_entropy(y, res), mydnn.calc_accuracy(y, res)

    @staticmethod
    def calc_accuracy(y, y_hat):
        y_arg_max, y_hat_arg_max = np.argmax(y, axis=-1), np.argmax(y_hat, axis=-1)
        return len(y_arg_max[y_arg_max == y_hat_arg_max]) / len(y)

    @staticmethod
    def calc_mse(y, y_hat):
        """
        :param y: The labels
        :param y_hat: The predictions
        :return: Calculated loss
        """
        return ((y - y_hat) ** 2).mean(axis=None)

    @staticmethod
    def calc_cross_entropy(y, y_hat):
        """
        :param y: The labels
        :param y_hat: The predictions
        :return: Calculated loss
        for each label p, the loss is calculated by: -sum(p[i] * log(p_hat[i]))
        """
        n = y.shape[0]
        return -np.sum(y * np.log(y_hat)) / n


def plot_graph(test_title, history, use_acc=True, save_name=None):
    val_acc = [history[i]['val_acc'] for i in range(len(history))]
    acc = [history[i]['acc'] for i in range(len(history))]
    val_loss = [history[i]['val_loss'] for i in range(len(history))]
    loss = [history[i]['loss'] for i in range(len(history))]
    steps = [history[i]['step'] for i in range(len(history))]

    if use_acc:
        major_y_ticks = np.arange(0, 1.1, 0.1)
        minor_y_ticks = np.arange(0, 1.02, 0.02)
        major_x_ticks = np.arange(0, max(steps) + 1, 5)
        minor_x_ticks = np.arange(0, max(steps) + 1, 1)

        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle(test_title, fontsize=10)
        axs[0].set_yticks(major_y_ticks)
        axs[0].set_yticks(minor_y_ticks, minor=True)
        axs[0].set_xticks(major_x_ticks)
        axs[0].set_xticks(minor_x_ticks, minor=True)
        axs[0].plot(steps, val_acc, label='val_acc')
        axs[0].plot(steps, acc, label='acc')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_title('Accuracy per epoch')
        axs[0].legend()
        axs[0].grid(which='both')

        major_y_ticks = np.arange(0, 5, 0.5)
        minor_y_ticks = np.arange(0, 5, 0.1)
        major_x_ticks = np.arange(0, max(steps) + 1, 5)
        minor_x_ticks = np.arange(0, max(steps) + 1, 1)
        axs[1].set_yticks(major_y_ticks)
        axs[1].set_yticks(minor_y_ticks, minor=True)
        axs[1].set_xticks(major_x_ticks)
        axs[1].set_xticks(minor_x_ticks, minor=True)
        axs[1].plot(steps, val_loss, label='val_loss')
        axs[1].plot(steps, loss, label='loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Loss per epoch')
        axs[1].legend()
        axs[1].grid(which='both')
    else:
        plt.figure(num=test_title)
        plt.title(test_title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.plot(steps, val_loss, label='val_loss')
        plt.plot(steps, loss, label='loss')
        plt.legend()
        plt.grid(which='both')

    if save_name is not None:
        save_name = path.join('graphs', save_name)
        plt.savefig("{}.png".format(save_name))

    # plt.show()
    plt.close()


def cook_data():
    if not path.exists("mnist.pkl.gz"):
        data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
        # Load the dataset
        urllib.request.urlretrieve(data_url, "mnist.pkl.gz")

    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    xt = mydnn.subtract_mean(train_set[0])
    yt = to_categorical(train_set[1])

    xv = mydnn.subtract_mean(valid_set[0])
    yv = to_categorical(valid_set[1])

    return xt, yt, xv, yv


def batch_size_experiments(epochs_):
    def get_test_name(arch_, batch_size_):
        res = 'batch_size_' + str(batch_size_)
        title_ = 'Batch size = {}\nArchitecture:\n'.format(str(batch_size_))
        for i, layer in enumerate(arch_):
            activation = layer['nonlinear']
            regularization = layer['regularization']
            input_shape = layer['input']
            output_shape = layer['output']

            res += '_' + activation + '_' + regularization
            title_ += '{}: Activation={}, Regularization={}, Input={}, Output={}\n'.format(str(i+1), activation, regularization, input_shape, output_shape)

        return res, title_

    arch_basic = [{'input': 784, 'output': 128, 'nonlinear': 'relu', 'regularization': 'l1'},
                  {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    arch_test_1 = [{'input': 784, 'output': 400, 'nonlinear': 'relu', 'regularization': 'l1'},
                   {'input': 400, 'output': 128, 'nonlinear': 'relu', 'regularization': 'l1'},
                   {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    arch_test_2 = [{'input': 784, 'output': 400, 'nonlinear': 'relu', 'regularization': 'l2'},
                   {'input': 400, 'output': 128, 'nonlinear': 'relu', 'regularization': 'l2'},
                   {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l2'}]

    arch_test_3 = [{'input': 784, 'output': 400, 'nonlinear': 'sigmoid', 'regularization': 'l2'},
                   {'input': 400, 'output': 128, 'nonlinear': 'sigmoid', 'regularization': 'l2'},
                   {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l2'}]

    arches = [arch_basic, arch_test_1, arch_test_2, arch_test_3]

    x_train, y_train, x_validation, y_validation = cook_data()

    batch_sizes = [128, 1024, 10000]
    for batch_size, arch in itertools.product(batch_sizes, arches):
        save_name, title = get_test_name(arch, batch_size)

        print("Testing:")
        print(title)
        full_save_path = path.join('graphs', save_name) + '.png'
        if path.exists(full_save_path):
            print('Skipping test because the graph already exists')
            continue

        nn = mydnn(arch, loss='cross-entropy')
        history = nn.fit(x_train,y_train,
                         epochs=epochs_,
                         batch_size=batch_size,
                         learning_rate=0.01,
                         x_val=x_validation,
                         y_val=y_validation)

        plot_graph(test_title=title,
                   history=history,
                   use_acc=True,
                   save_name=save_name)


def regularization_experiments(epochs_):
    def get_test_name(arch_, regularization_val_):
        res = 'regularization_value_{}'.format(regularization_val_)
        title_ = 'Regularization lambda = {}\n'.format(str(regularization_val_))
        for i, layer in enumerate(arch_):
            activation = layer['nonlinear']
            regularization = layer['regularization']
            input_shape = layer['input']
            output_shape = layer['output']

            res += '_' + activation + '_' + regularization
            title_ += '{}: Activation={}, Regularization={}, Input={}, Output={}\n'.format(str(i+1), activation, regularization, input_shape, output_shape)

        return res, title_

    arch_basic_l1 = [{'input': 784, 'output': 128, 'nonlinear': 'relu', 'regularization': 'l1'},
                     {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    arch_basic_l2 = [{'input': 784, 'output': 128, 'nonlinear': 'relu', 'regularization': 'l2'},
                     {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l2'}]

    arches = [arch_basic_l1, arch_basic_l2]
    regularization_values = [0, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 5e-3, 5e-2, 5e-1]

    x_train, y_train, x_validation, y_validation = cook_data()
    batch_size = 128

    for reg_val, arch in itertools.product(regularization_values, arches):
        save_name, title = get_test_name(arch, reg_val)
        print("Testing:")
        print(title)
        full_save_path = path.join('graphs', save_name) + '.png'
        if path.exists(full_save_path):
            print('Skipping test because the graph already exists')
            continue

        nn = mydnn(arch, loss='cross-entropy', weight_decay=reg_val)
        history = nn.fit(x_train, y_train,
                         epochs=epochs_,
                         batch_size=batch_size,
                         learning_rate=0.01,
                         x_val=x_validation,
                         y_val=y_validation)

        plot_graph(test_title=title,
                   history=history,
                   use_acc=True,
                   save_name=save_name)


def architecture_experiments(epochs_):
    def get_test_name(arch_):
        res = 'architecture'
        title_ = 'Architecture:\n'
        for i, layer in enumerate(arch_):
            activation = layer['nonlinear']
            regularization = layer['regularization']
            input_shape = layer['input']
            output_shape = layer['output']

            res += '_{}_{}_{}to{}'.format(activation, regularization, str(input_shape), str(output_shape))
            title_ += '{}: Activation={}, Input={}, Output={}\n'.format(str(i+1), activation, input_shape, output_shape)

        return res, title_

    '''
    ===================================================
                        Width = 512
    ===================================================
    '''
    test1 = [{'input': 784, 'output': 512, 'nonlinear': 'relu', 'regularization': 'l1'},
             {'input': 512, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]
    test2 = [{'input': 784, 'output': 512, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
             {'input': 512, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    '''
    ===================================================
                        Width = 360
    =================================================== 
    '''
    test3 = [{'input': 784, 'output': 360, 'nonlinear': 'relu', 'regularization': 'l1'},
             {'input': 360, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]
    test4 = [{'input': 784, 'output': 360, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
             {'input': 360, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    '''
    ===================================================
                        Width = 200
    =================================================== 
    '''
    test5 = [{'input': 784, 'output': 200, 'nonlinear': 'relu', 'regularization': 'l1'},
             {'input': 200, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]
    test6 = [{'input': 784, 'output': 200, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
             {'input': 200, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    '''
    ===================================================
                        Width = 512
                        1 hidden layer
    =================================================== 
    '''
    test7 = [{'input': 784, 'output': 512, 'nonlinear': 'relu', 'regularization': 'l1'},
             {'input': 512, 'output': 200, 'nonlinear': 'relu', 'regularization': 'l1'},
             {'input': 200, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]
    test8 = [{'input': 784, 'output': 512, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
             {'input': 512, 'output': 200, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
             {'input': 200, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    '''
    ===================================================
                        Width = 360
                        1 hidden layer
    =================================================== 
    '''
    test9 = [{'input': 784, 'output': 360, 'nonlinear': 'relu', 'regularization': 'l1'},
             {'input': 360, 'output': 128, 'nonlinear': 'relu', 'regularization': 'l1'},
             {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]
    test10 = [{'input': 784, 'output': 360, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
              {'input': 360, 'output': 128, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
              {'input': 128, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    '''
    ===================================================
                        Width = 200
                        1 hidden layer
    =================================================== 
    '''
    test11 = [{'input': 784, 'output': 200, 'nonlinear': 'relu', 'regularization': 'l1'},
              {'input': 200, 'output': 80, 'nonlinear': 'relu', 'regularization': 'l1'},
              {'input': 80, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]
    test12 = [{'input': 784, 'output': 200, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
              {'input': 200, 'output': 80, 'nonlinear': 'sigmoid', 'regularization': 'l1'},
              {'input': 80, 'output': 10, 'nonlinear': 'soft-max', 'regularization': 'l1'}]

    arches = [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12]
    x_train, y_train, x_validation, y_validation = cook_data()
    batch_size = 128

    for arch in arches:
        save_name, title = get_test_name(arch)
        print("Testing:")
        print(title)
        full_save_path = path.join('graphs', save_name) + '.png'
        if path.exists(full_save_path):
            print('Skipping test because the graph already exists')
            continue

        nn = mydnn(arch, loss='cross-entropy')
        history = nn.fit(x_train, y_train,
                         epochs=epochs_,
                         batch_size=batch_size,
                         learning_rate=0.01,
                         x_val=x_validation,
                         y_val=y_validation)

        plot_graph(test_title=title,
                   history=history,
                   use_acc=True,
                   save_name=save_name)


def three_dimension_plot(test_title, x_range, y_range, neural_network: mydnn, real_function, save_name=None):
    fig = plt.figure()
    fig.suptitle(test_title, fontsize=10)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    x, y = np.meshgrid(x_range, y_range)
    x_reshaped, y_reshaped = x.reshape(-1), y.reshape(-1)
    network_input = np.array(list(zip(list(x_reshaped), list(y_reshaped))))
    z = neural_network.predict(network_input, 10000).reshape(x.shape)
    z_hat = np.array([real_function(*x) for x in network_input]).reshape(x.shape)
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x, y, z_hat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    if save_name is not None:
        save_name = path.join('graphs', save_name)
        plt.savefig("{}.png".format(save_name))
    plt.close()

def regression_experiments():
    def f(x1, x2):
        return x1 * np.exp(-(x1 ** 2) + -(x2 ** 2))

    def get_test_name(arch_, m):
        title_ = 'm = {}\n'.format(str(m))
        for i, layer in enumerate(arch_):
            activation = layer['nonlinear']
            regularization = layer['regularization']
            input_shape = layer['input']
            output_shape = layer['output']

            title_ += '{}: Activation={}, Regularization={}, Input={}, Output={}\n'.format(str(i+1), activation, regularization, input_shape, output_shape)

        return title_

    m = 100
    first_samples = random.sample(list(np.linspace(-2,2,1000)), m)
    second_samples = random.sample(list(np.linspace(-2,2,1000)), m)
    x_train = np.array(list(zip(first_samples, second_samples)))
    y_train = np.array([[f(x1,x2)] for x1,x2 in zip(first_samples, second_samples)])
    arch = [{'input': 2, 'output': 64, 'nonlinear': 'relu', 'regularization': 'l2'},
            {'input': 64, 'output': 1, 'nonlinear': 'none', 'regularization': 'l2'}]

    nn = mydnn(arch, loss="MSE")
    history = nn.fit(x_train, y_train,
                     epochs=10000,
                     batch_size=100,
                     learning_rate=0.05,
                     learning_rate_decay=0.999,
                     min_lr=0.00001)
    three_dimension_plot(get_test_name(arch, m), np.linspace(-2,2,1000), np.linspace(-2,2,1000), nn, f, "regression_{}".format(m))

    m = 1000
    first_samples = random.sample(list(np.linspace(-2,2,1000)), m)
    second_samples = random.sample(list(np.linspace(-2,2,1000)), m)
    x_train = np.array(list(zip(first_samples, second_samples)))
    y_train = np.array([[f(x1,x2)] for x1,x2 in zip(first_samples, second_samples)])
    arch = [{'input': 2, 'output': 64, 'nonlinear': 'relu', 'regularization': 'l2'},
            {'input': 64, 'output': 1, 'nonlinear': 'none', 'regularization': 'l2'}]

    nn = mydnn(arch, loss="MSE")
    history = nn.fit(x_train, y_train,
                     epochs=10000,
                     batch_size=100,
                     learning_rate=0.05,
                     learning_rate_decay=0.999,
                     min_lr=0.00001)
    three_dimension_plot(get_test_name(arch, m), np.linspace(-2,2,1000), np.linspace(-2,2,1000), nn, f, "regression_{}".format(m))
    test_name = "regression test"

    plot_graph(test_title=test_name,
               history=history,
               use_acc=False,
               save_name=test_name)


if __name__ == '__main__':
    # np.seterr(all='raise')
    if not path.exists('graphs'):
        os.mkdir('graphs')

    epochs = 40
    batch_size_experiments(epochs)
    regularization_experiments(epochs)
    architecture_experiments(epochs)
    regression_experiments()
