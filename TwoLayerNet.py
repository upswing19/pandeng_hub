"""
    两层神经网络
        1. 结构 （input_size, hidden_size, output_size）
            定义初始化
                （1）. 根据结构初始化权重和偏置
                （2）. 根据结构初始化层
            定义预测函数
            定义损失函数
            定义梯度函数


"""
from utils.Affine_nn import Affine
from utils.ReLU_nn import Relu
from utils.Softmax_with_loss import SoftWithLoss
from collections import OrderedDict
import numpy as np


class TwoLayerNet:
    def __init__(self):
        # 权重的初始值,用标准差是0.01的高斯分布代替
        self.params = {}
        self.params['W1'] = np.array([[0.1, 0.3, 0.4], [0.2, 0.5, 0.6]])
        self.params['b1'] = np.array([0.5, 0.1, 0.2])
        self.params['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.params['b2'] = np.array([0.1, 0.2])

        # 初始化层
        self.layers = OrderedDict()  # 有顺序的字典
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 为什么不把最后一层存到字典中，因为最后一层的激活函数往往根据神经网络的功能来确定
        # 而且在进行预测的时候不需要激活函数
        self.lastLayer = SoftWithLoss()

    def predict(self, x):
        # x是输入数据，最终返回得分
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        # a1 = self.layers['Affine1'].forward(x)
        # z1 = self.layers['ReLU1'].forward(a1)
        # a2 = self.layers['Affine2'].forward(z1)

    def loss(self, x, t):
        # x是输入数据，t是标记
        predict = self.predict(x)
        loss = self.lastLayer.forward(predict, t)
        return loss # 四个数据的损失总和

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # 函数的参数axis表示沿哪个轴搜索最大值。默认情况下，axis=1，即沿第二个轴（列）搜索最大值。
        # 如果数据的组织方式与默认情况相反，可以使用axis=0或axis=-1来调整搜索方向。
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        """
        计算梯度
        :param x: 原始数据
        :param t: 原始数据对应的标签
        :return: 当前loss关于权重W和偏置b的梯度
        """
        # forward
        # 传播到最后
        self.loss(x, t)

        # backward
        d_soft = self.lastLayer.backward()
        print("d_soft: ", d_soft)
        d_x2, d_W2, d_b2 = self.layers['Affine2'].backward(d_soft)
        print("d_x2: ", d_x2)
        print("d_W2: ", d_W2)
        print("d_b2: ", d_b2)

        d_relu = self.layers['ReLU1'].backward(d_x2)
        d_x1, d_W1, d_b1 = self.layers['Affine1'].backward(d_relu)

        grads = {'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2}

        return grads


if __name__ == '__main__':
    # 测试
    net = TwoLayerNet()

    print("net.params['W1']: ", net.params['W1'],net.params['W1'].shape)
    print("net.params['b1']: ", net.params['b1'],net.params['b1'].shape)
    print("net.params['W2']: ", net.params['W2'],net.params['W2'].shape)
    print("net.params['b2']: ", net.params['b2'],net.params['b2'].shape)

    x_train = np.array([[0.6, 0.9], [0.3, 0.1], [0.2, 0.4], [0.4, 0.6]])
    # 输入的一定是矩阵，而不能是向量，因为后面会有矩阵运算，矩阵转置是两个维度的变化，
    # 而在计算机中向量的转置只是一个维度的变化，所以（3，）转置后不会变为（1，3）
    x_label = np.array([[1.0, 0], [0, 1], [0, 1], [1, 0]])
    print("x_train_shape,x_label_shape: ", x_train.shape,x_label.shape)

    predict = net.predict(x_train) # 四个输入数据的得分
    print(f"四个数据的预测值: {predict}")
    print("predict_shape",predict.shape)

    loss = net.loss(x_train, x_label)
    print(f"损失值: {loss}")

    grads = net.gradient(x_train, x_label)
    print(f"W1的梯度: {grads['W1']}")

    print(f"b1的梯度: {grads['b1']}")

    print(f"W2的梯度: {grads['W2']}")

    print(f"b2的梯度: {grads['b2']}")

    # 梯度更新
    net.params['W1'] -= 0.01 * grads['W1']
    net.params['b1'] -= 0.01 * grads['b1']

    net.params['W2'] -= 0.01 * grads['W2']
    net.params['b2'] -= 0.01 * grads['b2']
