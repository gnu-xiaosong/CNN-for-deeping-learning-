"""
desc: ANN模型

难点： 1.W，B的矩阵数据定义形式  ******* 以及W和B更新形式(W可采用list存储，每一维度为一个layer层， B采用二维矩阵形式）ok
      2.数据集的载入方式  ok
      3.反向传播求解梯度   no
      4.编码性能优化：主要由于采用了粒度化的神经元实体对象的单一循环计算，这里导致了不能使用矩阵形式GPU加速计算，因素计算性能比较低  no
      5.正则优化解决过度拟合问题 ok
      6.多分类softmax激活函数的计算，和rule激活函数函数 no
"""
import pickle
import time
from datetime import datetime
from joblib import Parallel, delayed
import numpy as np
import layers
import plot_model
from Conv import Conv, Pooling, initialize_weights, convolve, flip_3d_kernel, convolve1
from batchIterator import BatchIterator
from gradientDescent import GradientDescentAlgorithm
from tools import check_dimension


class AnnModel:
    """
    desc: 神经网络模型
    """
    def __init__(self):
        """初始化操作"""
        self.m = 0
        self.validation_set = None

    def Sequential(self, layer_list=[
        Conv(),
        Pooling(),
        layers.Dense(),
        layers.Dense(),
    ]):
        """
        desc: 定义模型处理流程顺序 前向传播流程  Hidden layer 配置
        parmeters:
            layer_list =    [
                  Dense(),
                  Dense()
                  .....
                ]
        """
        pre_layer = []
        layer_fc  = []

        # 网络结构
        self.Layers = layer_list

        # 拆分网络结构FC
        for layer in layer_list:
            # 判断是否为全连接层
            if isinstance(layer, list):
                layer_fc.append(layer)
            else:
                pre_layer.append(layer)

        # 前置层网络
        self.pre_layers = pre_layer
        # 全连接层网络
        self.layers     = layer_fc



    def init_kernal(self, pre_layers):
        """
        desc: 初始化卷积核权重参数
        """
        if not pre_layers:
            return True

        kernel_w = []
        # 遍历前置层
        for layer in pre_layers:
            w      = layer.f
            h      = layer.f
            channel = layer.n_c
            weight_shape = (w, h, channel)  # 权重的形状
            weights = initialize_weights(weight_shape)  # 初始化权重

            kernel_w.append(weights)


        print("--------------kernel weights--------------------")
        print(kernel_w)

        return kernel_w


    def compiler(self, lossType='mean_squared_error', gradientAlgorithm="batch_gradient_descent", miniBatchSize=10, lr = 0.01, _lambda=0, Y_classify=[],
                 validation_splite=0.2,
                 patience=100
    ):
        """
        desc: 针对定义好的模型在模拟和前的处理操作：
              比如向量化W，b，X，等前置操作，便于模型训练拟合
              adem优化等操作都在这里
              损失函数类型定义等
              loss     str     采用的损失函数类型：
                        均方差              mean_squared_error
                        稀疏交叉熵（多分类）   sparse_categorical_crossentropy
                        交叉熵(二分类)       cross_entropy_loss
              _lambda  float   正则化系数 默认0，只惩罚W模型参数
              lr      float    学习率
              Y_classify   list 分类类别标签 注意索引一一对应   分类问题的参数，如果为空 二分类问题则用0 1表示类别 如果为多分类或多标签则 依次用 0 1 2.。。表示类别
              gradientAlgorithm   str 梯度下降算法类型:  batch_gradient_descent(默认)、mini_batch_gradient_descent、stochastic_gradient_descent
              miniBatchSize   int  mini batch梯度下降的batch大小 默认10个
               validation_splite float   验证集占比例  默认0.2
        """

        # 根据Sequential处理流程获取网络模型结构以初始化W，b 调用init_wb方法

        # 设置损失函数类型
        self.lossType = lossType
        # 设置正则系数
        self._lambda = _lambda
        # 设置学习率
        self.lr = lr
        # 梯度下降算法类型
        self.gradientAlgorithm = gradientAlgorithm
        # mini-batch size大小
        self.miniBatchSize = miniBatchSize
        # 设置分类类别Y的标签
        self.Y_classify =  Y_classify
        # 设置耐心值，即连续几个 epoch 验证集性能没有提升就停止训练
        self.patience=patience
        # 验证集所占比例
        self.validation_splite = validation_splite
        # 其他一些优化操作：如Adam自学习率优化


    def  data_dispose(self, data_set):
        """
        desc: 数据集预处理
        """

        data_set_dispose=data_set

        return data_set_dispose


    def init_wb(self, layers, feature_count=0):
        """
        desc: 初始化模型参数W、b
        提示：模型参数不能全部初始化为0，逻辑回归和线性回归可以，因为根据递推公式知偏导数除了每层的偏置项外其余节点都为同一个，相当于在重复计算同一个结果，一堆特征变成了一个特征（虽然有很多单元，但值都相等）。
        故相当于仅有一个特征，这样神经网络的性能和效果就会大大折扣，因此需要进行随机初始化。
        随机初始化作用：有效防范梯度爆炸或消失，不同的初始化方式根据激活函数而定，以下为常见的激活函数Relu的初始化
        paremeters:
                feature_count   特征数量
                layers  list
                 >>[
                  Dense(),
                  Dense()
                  .....
                ]
        return:
            W list  初始化参数
            B list  初始化参数
        """
        # 获取数据集的特征个数
        a_in_num = feature_count

        # 定义机构类型
        W = []
        B = []

        for l,layer in enumerate(layers):
            # 获取神经元个数
            neuron_num = len(layer)

            #--------------作为演示用：全部初始化为0时-----------
            # 初始化layer层的偏置b
            # b = np.zeros((neuron_num ,1), dtype=np.float64).reshape(-1, 1)
            # # 初始化: 其列项数应为前一层的a输入个数
            # w = np.zeros((neuron_num, a_in_num), dtype=np.float64)

            #----------------随机初始化:relu----------------------
            b = np.random.rand(neuron_num, 1).reshape(-1, 1)
            # 初始化: 其列项数应为前一层的a输入个数
            w = np.random.rand(neuron_num, a_in_num) * np.sqrt(2 / a_in_num)


            # print(f"b dtype={b.dtype} ")

            # append进入W，B数组中
            W.append(w),B.append(b)

            # 更新# 输入a的个数等于该层的w行数
            a_in_num =  int(w.shape[0])



        return W, B


    def learning_rate(self):
        """
        desc: 学习率
        """


        return  self.lr


    def data_set_splite(self, data_set, validation_split):
        """
        desc: 数据集划分=训练集和测试集

        paremeters:
                data_set=(data_X, data_Y)

        return:
             train_set,test_set =  {
            "X":矩阵  行为样本数  列为特征数
            "Y":  列向量
        }
        """
        # 样本数
        self.m += len(data_set[1])

        self.validation_splite_m = int(np.ceil(np.array(data_set[0]).shape[0] * validation_split))

        train_set = {
            "X":data_set[0][ self.validation_splite_m:,:],
            "Y":data_set[1][ self.validation_splite_m:]
        }

        test_set  = {
            "X": data_set[0][ :self.validation_splite_m,:],
            "Y": data_set[1][ :self.validation_splite_m]
        }


        return  train_set,test_set


    def classify_Y_dispose(self, Y):
        """
        desc: one_hot编码  多分类的Y值处理，将输入列向量Y转变成多类别的矩阵形式
        parameters:
                Y    numpy  列向量每个元素为对应类别
        return:
                Y_   numpy  矩阵  每行为一个样本，每列为对应输出层的真实值类别
        """
        Y = np.array(Y).reshape(-1, 1)
        num_classes = len(self.layers[-1])  # 输出层类别数
        Y_ = []

        # 定义要执行的计算函数
        # def compute_y_i(class_index, num_classes):
        #     return [1 if i == class_index else 0 for i in range(num_classes)]
        # # 并行计算
        # Y_ = Parallel(n_jobs=1)(delayed(compute_y_i)(class_index, num_classes) for class_index in Y[:, 0])


        # # 遍历每一个Y元素类别
        for class_index in Y[:, 0]:
            # 设置每个样本对应的类别
            y_i = [1 if i == class_index else 0 for i in range(num_classes)]
            Y_.append(y_i)


        return np.array(Y_)



    def save_model_paremeters(self, test_sample_count=0,
                              test_mae=0, test_correct_rate=0, name=None):
        """
        desc: 保存模型参数
        """

        paremeters = {
            # 模型训练名
            "name": name,
            # 网络结构
            "layers":self.Layers,
            # 学习率
            "learning_rate": self.learning_rate(),
            # 迭代次数
            "epochs": self.epochs,
            # 数据集样本数：训练集+验证集
            "sample_count": self.m,
            # 对应标签labels
            "Y_labels": self.Y_classify,
            # 损失函数类型
            "lossType": self.lossType,
            # 正则参数
            "regularize_lambda": self._lambda,
            # 验证集占比
            "validation_splite": self.validation_splite,
            # 设置耐心值，即连续几个 epoch 验证集性能没有提升就停止训练
            "patience": self.patience,
            # 训练时间
            "train_time": self.second_enmu_iteration * self.epochs,
            # 梯度下降算法
            "gradientAlgorithm": self.gradientAlgorithm,
            # mini_batch 算法的batch size大小
            "miniBatchSize":self.miniBatchSize,
            # 对应Y的索引类别
            "Y_classify": self.Y_classify,
            # 最小损失
            "min_cost": min(self.Cost),
            # 最佳迭代次数
            "best_epoch": self.best_epoch,
            # 验证集表现
            "validation_performance": {
                # 验证集样本数
                "sample_count": self.validation_splite_m,
                # 均对误差
                "MAE": self.best_performance,
                # 准确率
                "correct_rate":self.best_accuracy,
            },
            # 测试表现
            "test_performance":{
                # 验证集样本数
                "sample_count": test_sample_count,
                # 均对误差
                "MAE": test_mae,
                # 准确率
                "correct_rate": test_correct_rate,
            },
            # 模型参数W
            "W": self.W_opt,
            # 模型参数
            "B": self.B_opt
        }
        # 获取当前时间
        current_time = datetime.now()
        # 将时间格式化为字符串，作为文件名
        file_name ="model_"+ str(name) +"_test="+str(test_correct_rate)+"_|_"+str(test_mae)+"_"+ current_time.strftime("%Y_%m_%d_%H_%M_%S")+"_pickle_byte"
        with open('./models/'+file_name, 'wb') as file:
            pickle.dump(paremeters, file)



    def mutiple_classify_Y_dispose(self, data_set):
        """
        desc:针对多分类、多标签分类的Y进行ont-hat编码处理
        paremeters:
                data_set  dict 数据集
                 {
                     "X":,
                     "Y":
                 }
        return:
            返回经过处理数据集data_set
        """
        # 判断是否是多分类问题时
        if self.lossType == "sparse_categorical_crossentropy" and np.array(data_set['Y']).shape[1] == 1:
            # 多分类问题的Y矩阵形式处理
            data_set['Y'] = self.classify_Y_dispose(data_set['Y'])

            # 判断输出单元个数是否与Y的类别匹配
            if len(self.layers[-1]) != data_set['Y'].shape[1]:
                print(f"输出单元个数:{len(self.layers[-1])} Y的类别个数: {data_set['Y'].shape[1]} 两者不匹配!")
                exit()
            return data_set

        else:
            return data_set


    def data_iterator(self, iterCall=None,  nextCall=None):
        """
        desc: 数据集自定义迭代器iterCall(), 具体见BatchIterator类中的iter和next方法， 对应
        iterCall: 参数为迭代对象 不用返回任何数据，只需要对其迭代器对象进行操作即可，该方法仅调用一次，在next之前
        nextCall 参数为迭代对象  返回要如下数据：对每个list的数组进行操作与处理
                 {
                    "X": dnumpy  矩阵  特征矩阵
                    "Y": dnumpy  列向量 对应类别索引
                  }
        """
        self.iterCall = iterCall
        self.nextCall = nextCall



    def fit(self, data_set, epochs=5, feature_count=None):
        """
        desc: 模型训练拟合
        paremeters:
            data_set  任何数据类型 | 字典  由feature_count控制  如果feature_count传入值不为None，则采用自定义数据集迭代器进行处理 类型为数组类型
                      需要对其数据进行处理需要在iterCall()回调函数中自定义组织成数组形式
            @任何数据类型:    每个元素为样本集合或单个样本，具体如何组织，需要自己编写迭代器nextCall函数
            @字典:    如下格式
                      {
                        "X": dnumpy  矩阵  特征矩阵
                        "Y": dnumpy  列向量 对应类别索引
                      }
            epochs    int             迭代次数
        """

        # 初始化模型参数W，和卷积核参数
        feature_count_tmp = feature_count if feature_count else data_set["X"].shape[1]
        W, B = self.init_wb(self.layers, feature_count_tmp)
        # 初始化卷积核
        kernel_w = self.init_kernal(self.pre_layers)

        # 存储迭代的损失
        self.Cost = []

        # 实例化gradientDescent
        GD = GradientDescentAlgorithm(ann=self, type=self.gradientAlgorithm, batch=self.miniBatchSize)
        # 计算每次迭代的单位运行时间: s/iteration
        # 记录开始时间
        start_time = time.time()

        self.best_accuracy = 0.0              # 初始化最佳正确率
        self.best_performance = float('inf')  # 初始化最佳性能，对于回归问题设置为正无穷
        self.best_epoch = 0                   # 初始化最佳迭代次数
        self.epochs = epochs
        for epoch in range(epochs):
            """训练轮回数"""
            # 判断是否采用数据集迭代对象操作

            if feature_count:
                iteration_data = BatchIterator(data_set, self.iterCall, self.nextCall)
            else:
                iteration_data = [ data_set ]

            for batch_data_set in  iteration_data:
                """
                desc:分批次数据加载模型训练: 使用自定义迭代器分批次加载训练数据
                paremeters:
                    batch_data_set  dict 每次从迭代器中获取的数据集
                    batch_data_set训练集
                     {
                         "X":,
                         "Y": 还未one-hat编码的列向量
                     }
                """

                # print(batch_data_set)
                # 维度监测
                if not check_dimension(batch_data_set['Y'], 2, column_vector=True):
                    print(f"warnig: model.py fit() batch_data_set['Y'] 维度不符合! 应该为列向量！已自动转化为列向量")
                    # 自动转化为列向量
                    batch_data_set['Y'] = np.array(batch_data_set['Y']).reshape(-1, 1)


                # 划分验证集与训练集的比例
                train_set, self.validation_set = self.data_set_splite((np.array(batch_data_set["X"]).copy(), np.array(batch_data_set["Y"]).copy() ),
                                                                      validation_split=self.validation_splite)
                # 数据集one-hat编码
                train_set = self.mutiple_classify_Y_dispose(train_set)

                # 调用梯度下降算法类方法实现
                W, B, kernel_w = GD.gradient_descent(W, B,kernel_w, epoch=epoch, train_set=train_set)


                # 在验证集上评估模型性能
                validation_performance = self.evaluate(self.validation_set, W, B)[0]
                print(f"Epoch {epoch + 1}, validation_performance: {validation_performance}")

                if self.lossType=="mean_squared_error":
                    # 回归问题: 如果验证集上的性能优于之前的最佳性能，则更新最佳性能和最佳迭代次数
                    if validation_performance < self.best_performance:
                        self.best_performance = validation_performance
                        self.best_epoch = epoch
                        # 更新最优参数
                        self.W_opt = W
                        self.B_opt = B
                else:
                    # 分类问题: 如果验证集上的准确率优于之前的最佳准确率，则更新最佳准确率和最佳迭代次数
                    if validation_performance > self.best_accuracy:
                        self.best_accuracy = validation_performance
                        self.best_epoch = epoch
                        # 更新最优参数
                        self.W_opt = W
                        self.B_opt = B
            # 检查是否满足停止迭代条件（例如性能连续几个 epoch 都没有提升）
            # patience = self.patience     # 设置耐心值，即连续几个 epoch 验证集性能没有提升就停止训练
            if epoch - self.best_epoch >= self.patience:
                print(f"best_epoch={self.best_epoch} self.best_performance={self.best_performance} self.best_accuracy={self.best_accuracy} No improvement in validation performance for {self.patience} epochs. Stopping training.")
                break

            # 如果迭代次数不够，即还没到最优解，则把最后迭代W，B赋值为W_opt、B_opt
            if self.best_epoch==epochs:
                self.W_opt = W
                self.B_opt = B

        # 记录结束时间
        end_time = time.time()
        # 计算单位迭代次数所用时间
        self.second_enmu_iteration = (end_time-start_time) / epochs

        # 绘制损失函数
        plot_model.plot_cost(self.Cost)

        return self.W_opt, self.B_opt, self.Cost[-1]


    def forward(self, data_set, W, B, kernal):
        """
        desc: 前向传播算法，用于求解样本一次经过神经网路的输出值
        paremeters:
            data_set={
            "X":,
            "Y":
        }
            W      list  按次序每个item元素为一层layer
            B      list  数组  每item为对应一层layer的偏置
            kernal  list 卷积核的权重  每个元素对应每层的卷积核权重
        return:
            a_out  matrix  每列为一个样本的预测值y_hat
            Z      list   每个item元素为对应层layer的Z，列向量对应得到的神经元的Z值 Z = [item_1, item_2...item_i...item_L]    item_l = [z_l1, z_l2...z_li..z_lm]  z_li为列向量对应l层的第i个样本神经元的z
            A      list   每个item元素为对应层layer的A，列向量对应得到的神经元的A值 A = [item_1, item_2...item_i...item_L]    item_l = [a_l1, a_l2...a_li..a_lm]  a_li为列向量 对应l层的第i样本神经元的a
        """
        # 存储计算的Z值和A值： A与Z一一对应的
        Z = []  # 每个item元素为对应层layer的Z，列向量对应得到的神经元的Z值 Z = [item_1, item_2...item_i...item_L]    item_l = [z_l1, z_l2...z_li..z_lm]  z_li为列向量对应l层的第i个样本神经元的z
        A = []  # 每个item元素为对应层layer的A，列向量对应得到的神经元的A值 A = [item_1, item_2...item_i...item_L]    item_l = [a_l1, a_l2...a_li..a_lm]  a_li为列向量 对应l层的第i样本神经元的a

        # 2.前向传播 计算Y_hat  预测值矩阵形式
        ## 前置层网络操作：卷积层、池化层
        data_pre = data_set.copy()

        conv_Z = []
        conv_A = []
        a_conv_0 = []
        for layer_n, pre_layer in enumerate(self.pre_layers):
            # 计算卷积操作后的data、z、a值
            data_pre, z, a,a_conv_0  = pre_layer.run(data_pre, kernal[layer_n])
            # 计算存储Z
            conv_Z.append(z)
            # 计算存储A
            conv_A.append(a)


        print(f"卷积层计算过后: data_pre={data_pre}")
        ## （1）FC全连接层网络: 初始化A为数据集特征输入: a_(0) = x_(i) 每列为样本，对应列值a为l层的a值，默认输出层a_0不append进入A中
        # a_in = np.array(data_set["X"]).T  # 这里转置了,输入层 为列向量
        a_in = np.array(data_pre["X"]).T  # 这里转置了,输入层 为列向量

        # 遍历更新递归输入
        a = a_in

        ## （2）预测值
        for layer_n in range(len(W)):
            """
            desc:一层layer所做的工作
            具体包括：
                1.计算模型值z 
                2.计算激活值a  遍历作为下一层的输入
            """


            # 线性模型: 所得到的z_l = [z_l1, z_l2...z_li..z_lm]  所有样本的l层的z值
            z = self.layers[layer_n][0].Z(a, W[layer_n], B[layer_n])
            Z.append(z)  # 存储layer_n层的z值

            # 激活:更新A作为输入又作为输出
            a = self.layers[layer_n][0].A_out(z)
            A.append(a)  # 存储layer_n层的a值

        # 输出层的a_out = [a_L1, a_L2...a_Li..a_Lm]  每列为一个样本的输出为一个列向量y_hat=a_L=a_out
        a_out = a  # 为一个列向量


        return a_out, Z, A, conv_Z, conv_A,a_conv_0

    def backward(self, data_set, W, B, kernel_w):
        """
        desc: 反向传播算法，求偏导
        所需数据准备：
            1.数据集：data_set = {
                "X": ,
                "Y":
            }
            单个样本：{x_(i), y_(i)}
            2.计算各层的输出a_(l)
            3.真实值：y_(i)
            4.l层的权重参数W_(l)、l-1层Z_(l-1)
             ----->最终要得到：a_(l) * (l+1层的误差项，即对Z_(l)的偏导)

             以上作为前提步骤：通过前向传播计算获得： W、a、Z
        算法步骤：
            For i to m:
                1.set a_0 = x_(i)
                2.perform forward propagation to compute  a_(l) for l=1、2、3....L
                3.Using y_(i),compute u_(l) = a_(l) -y_(i) 最后一项误差即对最后一层的J对Z的偏导
                4.迭代关系式反向传播计算误差即J对各自层Z的偏导
                5.累计偏导误差得到最后的对参数的偏导

            计算最终的偏导，加上正则项即得到最终的偏导值

            利用梯度关系实现参数的更新


        paremeters:
            data_set={
                "X":,
                "Y":
            }
            W  list    按次序每个item元素为一层layer
            B  list    每项为对应一层layer的偏置
            kernel_w list 对应卷积层的卷积核的权重
        return:
            dJ_dW_sum  list   与模型参数W一一对应
            dJ_dB_sum  list   与模型参数B一一对应
            dJ_dW_sum_conv: 卷积层总的权重梯度列表
        """
        # 样本数
        m = len(data_set["Y"])
        #神经网络层数：x输出层不算a_0
        L = len(W)
        # 卷积层数，包括池化层
        conv_layer_l = len(kernel_w)

        # 总体样本的误差(即偏导数)的总和:这里与W和B的形式刚好对应上,形式完全一样，全部初始化为0
        dJ_dW_sum = [ np.array(layer_W ) * 0  for layer_W in W]
        dJ_dB_sum = [ np.array(b) * 0 for b in B]
        dJ_dW_sum_conv = [ kernel * 0 for kernel in kernel_w]

        # print(f"W= {W} B={B}")


        for i in range(m):
            """
            desc: 循环遍历样本进行单个样本的偏导计算，再累计每个样本的偏导值得出最总的损失函数对W和b的偏导值
            """
            #设置第i个x样本（x_i, y_i)
            x = np.array(data_set["X"])[i,:]
            y = np.array(data_set["Y"])[i,:]

            # 1.set a_(0) = x_(i) 为列向量
            a_0 = np.array(x).reshape(-1, 1)

            # 2.计算前向传播计算各层的a_(l) for  l =1, 2,3....L  调用前向传播算法:self.forward()
            ## 封装data_set
            data_ = {
                "X": np.array([x]) ,
                "Y": np.array([y])
            }

            a_out, z, a,conv_z, conv_a, a_conv_0 =self.forward(data_, W, B)


            # 为一个列向量
            y = np.array([y]).T
            # 3.输出层：计算的误差和对W和b的偏导值：
            dJ_dz = np.array(a_out - y)                                         # J对z_(L)的偏导--"数"   为列向量

            # 单个样本的dJ_dz的数组， 用于存储各层的损失误差dJ_dz，每个item为层的dJ_dz，层数为倒序的dJ_dZ =  [L的dJ_dz, L-1的dJ_dz,...,1的dJ_dz]
            dJ_dZ = []
            dJ_dZ.append(dJ_dz)

            # print(f"|------------样本数={i+1}---------|")
            # print(f" a = {a}")
            # print(f"最后一层J对z偏导: dJ_dz_L={dJ_dz}")

            # 4.隐藏层：利用递归关系式求向后传播的误差(偏导数)   采用循环遍历(0开始索引)：L-2 to 0  只需要遍历到隐藏层的第2层
            for l in range(L-1, -1, -1):
                """ l层
                desc: 神经网络反向递归求解单个样本的损失函数loss对z偏导值，
                      再根据递推关系反向传播求解出神经网络中全部的dJ_dz值
                      同时求得J对W和b的偏导值dJ_dw、dJ_db
                       为从倒数第二层l-1层开始递减: l-1、l-2、....、1
                                    数组索引递减: l-2、l-3、....、0
                                    其中 l为数组中对应层的数组索引
                """
                dJ_dw = dJ_dz @ np.transpose(a_0 if l == 0 else a[l - 1])  # J对w_(L)的偏导--"数" 这里判断是否为第一层，是，则a[l-1]=a_0，否则按照a[l-1]
                dJ_db = np.array(dJ_dz).reshape(-1, 1)  * 1                # J对b_(L)的偏导--"数"

                # 梯度检测
                # print(f"layer ={l+1} 层, gradient real: {dJ_dw[0]}")
                # print(f"--------所在层数={l+1}---------")
                # print(f"dJ_dz={dJ_dz} dJ_dw={dJ_dw} dJ_db={dJ_db}")

                dJ_dW_sum[l] = np.array(dJ_dW_sum[l]) + dJ_dw  # 增加进对应层的总和dJ_dw
                dJ_dB_sum[l] = np.array(dJ_dB_sum[l]).reshape(-1,1) + dJ_db  # 增加进对应层的总和dJ_dw


                # 递归更新l-1层的误差dJ_dz
                if l !=0:
                    g_z_greadient = self.layers[l - 1][0].g_derivative(z[l - 1])  # 递归关系推导g_z_greadient为列向量
                    dJ_dz = np.transpose(W[l]) @  dJ_dz * g_z_greadient
                    dJ_dZ.append(dJ_dz)

###########################################-------------重点----------##################################################################################

            # 5.卷积层参数更新: 紧接着误差项为: dJ_dz, dJ_db
            for conv_layer_n in range(conv_layer_l -1, -1, -1):
                # 前一层的kernel_n-1的a值
                z_pre =  a_conv_0 if conv_layer_n == 0 else conv_a[conv_layer_n-1]
                # a_pre值
                a_pre = self.pre_layers[conv_layer_l].compute_A(z_pre)
                # 计算当前层的dJ_dw，会用到dJ_dz
                layer = self.pre_layers[conv_layer_l]
                # 遍历该层的卷积层的所有卷积核的
                kernels = kernel_w[conv_layer_n]
                dJ_dws = []
                for kernel_n in range(len(kernels)):
                    """单个卷积核操作"""

                    # 对每一个卷积核进行操作
                    kernel = kernels[kernel_n]
                    # 进行卷积核的误差dJ_dz更新，
                    ## 1. 扩充误差dJ_dz
                    p = layer.f - 1
                    dJ_dz_ = np.pad(dJ_dz, ((0, 0), (p, p), (p, p)), mode="constant",
                           constant_values=layer.padding_data)
                    # 计算误差值
                    dJ_dz = convolve(dJ_dz_, flip_3d_kernel(kernel), stride=(1, 1), padding=(0, 0)) * layer.compute_active_gradient(z_pre)

                    # 卷积核的偏导
                    dJ_dw = convolve1(a_pre, dJ_dz)
                    dJ_dws.append(dJ_dw)
                    dJ_dW_sum_conv[conv_layer_n] = np.array(dJ_dW_sum_conv[conv_layer_n]).reshape(-1, 1) + dJ_dw  # 增加进对应层的总和dJ_dw

                # 计算该层kernel_n的偏导数
                dJ_dW_sum_conv[conv_layer_n] = np.array(dJ_dW_sum_conv[conv_layer_n]) + np.array(dJ_dws)   # 增加进对应层的总和dJ_dw


##############################################################################################################################


        # 计算损失项的偏导数dCost_dW
        dCost_dW       =  [ (1 / m) * layer for  layer in dJ_dW_sum]
        dRegularize_dW =  [ (1 / m) * self._lambda * w   for w in W ]

        # 卷积层
        dCost_dkernel_dW = [(1 / m) * layer for layer in dJ_dW_sum_conv]
        dRegularize_dkernel_dW = [(1 / m) * self._lambda * w for w in kernel_w]


        #计算J对W，b的偏导
        dJ_dW =  [ dCost_dW[l] + dRegularize_dW[l] for l in range(L)]
        dJ_dB =  [ (1 / m) * layer for  layer in dJ_dB_sum]

        # 计算卷积层
        dJconv_dW = [ dCost_dkernel_dW[l] + dRegularize_dkernel_dW[l] for l in range(conv_layer_l)]


        return  dJ_dW, dJ_dB,dJconv_dW


    def J_wb(self, data_set, W, B, kernel_w=None):
        """
        计算样本的平均成本
        paremeters:
            data_set={
            "X":,
            "Y":
        }
            W  list  按次序每个item元素为一层layer
            B  list  数组  每元素为对应一层layer的偏置
        """
        # 样本个数
        m = len(data_set["X"])


        # print("-------------forward for J_wb(self, data_set, W, B)--------------------------")
        ## （3） 获取最终神经网络输出结果: 每行为一个样本的神经网络的输出
        Y_hat = self.forward(data_set, W, B, kernel_w)[0]          # 一次前向传播求预测值，列向量，各层的z和a值，列为所有样本
        Y     = data_set["Y"]                            # 真实值，numpy矩阵


        # 3.计算平均样本函数 J = 损失 + 正则
        ##  （1） 计算cost
        cost       = self.cost(Y_hat, Y, m)
        ##  (2)  计算正则化项（防止过拟合，以惩罚参数W)
        regularize = self.regularize(self._lambda,m,W)

        J_wb = cost + regularize

        return  J_wb



    def regularize(self, _lambda, m, W):
        """
        desc: 计算正则化项
        paremeers:
            _lambda   float  正则参数
            m         int    样本个数
            W         list       神经网络的参数W
        return:
            regularize  float 计算的正则化项
        """

        regularize = 0

        for layer_w in W:
            # 转为dnumpy对象
            regularize += np.sum(np.array(layer_w)**2)

        regularize *= (_lambda) / (2*m)

        return regularize


    def loss(self,y_hat, y):
        """
        desc: 根据采用的损失函数类型求解单个样本的损失值
        """
        def mean_squared_error(y_hat, y):
            """
            desc：损失函数计算算法，均方差损失函数  适用于线性回归
            paremeters:
                y_hat    float   神经网络预测值， 列向量
                y        float   对应标签的真实值 列向量
            return:
                loss     float   产生的损失
            """
            loss = (1/2) * np.array(y_hat - y) ** 2

            return loss

        def cross_entropy_loss(y_hat, y):
            """
               desc：损失函数计算算法，交叉熵损失函数  适用于二分类
               paremeters:
                   y_hat    float   神经网络预测值
                   y        float   对应标签的真实值
               return:
                   loss     float   产生的损失
               """
            # 防止出现除以零的情况，将预测概率中的非零值替换为一个极小值
            epsilon = 1e-15
            y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

            loss = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

            return loss

        def sparse_categorical_crossentropy(y_hat, y):
            """
              desc：损失函数计算算法，稀疏交叉熵损失函数  适用于多分类
              paremeters:
                  y_hat    float   神经网络预测值
                  y        float   对应标签的真实值
              return:
                  loss     float   产生的损失
              """
            epsilon = 1e-15  # 避免除以0的情况
            y_pred = np.clip(y_hat, epsilon, 1 - epsilon)  # 避免取对数时的溢出

            # loss为一个行向量，每个元素为样本的损失
            loss = np.sum(-np.array(y).T * np.log(y_hat), axis=0)

            return loss

        # 1.判断使用的损失函数类型
        if self.lossType == "mean_squared_error":
            loss = mean_squared_error(np.array(y_hat).reshape(-1,1) , np.array(y).reshape(-1, 1))
        elif self.lossType == "cross_entropy_loss":
            loss = cross_entropy_loss(np.array(y_hat).reshape(-1,1) , np.array(y).reshape(-1, 1))
        elif self.lossType == "sparse_categorical_crossentropy":
            loss = sparse_categorical_crossentropy(y_hat, y)

        return  loss


    def cost(self, Y_hat, Y, m):
        """
        desc: 用于求解全体样本的平均损失值
        """

        # if self.lossType == "mean_squared_error":
        #     # 回归问题
        #     cost = (1 / (2 * m)) * np.sum(self.loss(Y_hat, Y))
        # elif self.lossType == "sparse_categorical_crossentropy" and self.lossType == "cross_entropy_loss":
        #     # 分类问题
        #     cost = (1 / m) * np.sum(self.loss(Y_hat, Y))

        cost = (1 / m) * np.sum(self.loss(Y_hat, Y))

        return cost


    def predict(self, data_set):
        """
        desc: 用于根据输入数据进行预测  利用已经训练的模型最优参数： W_opt、B_opt
        paremeters:
            data_set={
            "X":,
            "Y":
            }
        """
        predict_y = self.forward(data_set, self.W_opt, self.B_opt)[0].T
        predict_label = []


        if self.lossType=="sparse_categorical_crossentropy":

            num_rows = predict_y.shape[0]
            pre_arr = np.zeros_like(predict_y)  # 创建与 predict_class 相同形状的全零数组

            for i in range(num_rows):
                max_index = np.argmax(predict_y[i])  # 找到当前行最大值的索引
                pre_arr[i, max_index] = 1  # 将最大值位置设置为1


            # 获取对应的Y类别标签
            if self.Y_classify:
                predict_label = [self.Y_classify[np.argmax(row)] for row in pre_arr]
            else:
                # 不为空
                predict_label = pre_arr


        elif self.lossType=="cross_entropy_loss":
            # 决策阈值
            threshold = 0.5
            # 二分类
            predict_label_index = np.where(predict_y > threshold, 1, 0)

            predict_label = [ self.Y_classify[int(index)] for index in predict_label_index]



        return predict_y,predict_label


    def evaluate(self, test_data={}, W=[], B=[]):
        """
        desc: 用于评估模型: 利用self.test_set数据集来进行测试
        调用前向传播算法: self.forward()求解预测值
        paremeters:
            test_data dict 测试集
            {
                "X":  dnumpy 矩阵,
                "Y":  列向量  对应值或类别的索引
            }
        """
        # 判断是否传入了W和B
        if not W and not B:
            # 没有传入
            W = self.W_opt
            B = self.B_opt

        # 真实值
        Y  = np.array(np.array(test_data['Y']).reshape(-1, 1)).copy()

        # 针对多分类、多标签分类的Y进行ont-hat编码处理
        test_data = self.mutiple_classify_Y_dispose(test_data)


        # 判断评价类型
        if self.lossType=="mean_squared_error":
            """
            desc: 线性回归模型，采用均方根误差RMSE、平均绝对误差MAE
            """
            # 预测
            predict = np.array(self.forward(test_data, W, B)[0][0]).reshape(-1, 1)

            # 均方根误差RMSE
            RMSE = np.sqrt(np.mean((Y - predict)**2))
            # 平均绝对误差MAE
            MAE =  np.mean(np.abs(Y - predict))

            return MAE,RMSE,predict,Y

        elif self.lossType == "sparse_categorical_crossentropy":
            Y_class = np.array(test_data['Y'])
            predict_class = self.forward(test_data, W, B)[0].T

            num_rows = predict_class.shape[0]
            pre_arr = np.zeros_like(predict_class)  # 创建与 predict_class 相同形状的全零数组

            for i in range(num_rows):
                max_index = np.argmax(predict_class[i])  # 找到当前行最大值的索引
                pre_arr[i, max_index] = 1  # 将最大值位置设置为1

            # 统计错误个数
            err_count = np.sum(np.any(pre_arr != Y_class, axis=1))


            correct_count = len(predict_class) - err_count

            # 计算正确率
            correct_rate = correct_count / len(predict_class)

            # 获取对应的Y类别标签
            if self.Y_classify:
                predict_label = [self.Y_classify[np.argmax(row)] for row in pre_arr]
            else:
                # 不为空
                predict_label = pre_arr

            return correct_rate, predict_class,pre_arr, Y, correct_count, err_count,predict_label

        elif self.lossType=="cross_entropy_loss":
            """
            desc: 二分类类问题，测试集模型的准确率
            """
            # 预测
            predict = np.array(self.forward(test_data, W, B)[0][0]).reshape(-1, 1)

            # 决策阈值
            threshold = 0.5
            predict_result = []
            for i in range(len(predict)):
                if predict[i] > threshold:
                    predict_result.append(1)
                else:
                    predict_result.append(0)

            # 比较
            result = np.abs(np.array(predict_result) - Y.T)

            # 统计正确个数
            err_count = np.sum(result)
            correct_count = len(predict) - err_count

            # 正确率
            correct_rate = correct_count / len(predict)

            return correct_rate, correct_count, err_count, predict_result, predict, Y.T.tolist()


    def customize(self,fun):
        """
        desc:用户自定义函数
        paremeters: self 为该类的对象
        """
        fun(self)

