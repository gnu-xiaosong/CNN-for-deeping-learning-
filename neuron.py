import numpy as np

import activations_model


class Neuron:
    """
    desc: 这是神经元neuron实体对象
    抽象方法和属性：
    @属性：

    @方法：

    """

    def __init__(self, active_fun, pos=(1, 1)):
        """
        desc：神经元初始化
        """
        # 1.激活函数类型: 默认linear线性激活
        self.active_fun = active_fun
        # 2.神经元在网络中所在位置
        self.pos = pos


    def Z(self, A_in, W, b):
        """
        desc: 线性模型  Z(X) = W @ A_in + b
        paremeters:
              A_in     np.dnumpy     输入列向量
                        A_in = np.array([ [a1],
                                          [a2],
                                          ....
                                          [aj]
                                        ])

              W        np.dnumpy     行向量   W = np.array([[ w1, w2,.....]])
              b        列向量          偏置

        return:
              Z_wb     列向量          计算线性模型值Z(每一列为该层的z值,列数为样本数)
        """

        Z_wb = W @  A_in + b

        return Z_wb

    def A_out(self, Z_wb):
        """
        desc: 计算neuron神经元输出
        paremeters:
            Z_wb   float   计算线性模型值Z
        return:
            A      float    计算出来的值
        """

        A = self.g(Z_wb)

        return A

    def g(self, z):
        """
        desc: 激活函数
        """
        if self.active_fun == "sigmod":
            z = activations_model.sigmod(z)
        elif self.active_fun == "rule":
            z = activations_model.rule_function(z)
        elif self.active_fun == "softmax":
            z = activations_model.softmax(z)
        else:
            z = z

        return z

    def g_derivative(self,x):
        """
        激活函数求导
        """
        if self.active_fun == "sigmod":
            gradient = activations_model.sigmod_derivative(x)
        elif self.active_fun == "rule":
            gradient = activations_model.rule_derivative(x)
        elif self.active_fun == "softmax":
            gradient = activations_model.softmax_derivative(x)
        else:
            gradient = 1

        return gradient