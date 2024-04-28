import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
from pylab import mpl
from package_xskj_NetworkXsimple import netGraph
from layers import Dense
from model import AnnModel

# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
sns.set_theme()
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体

# 实例化netGraph对象
networkGraph =netGraph(type=1)



# 加载数据集
data_set = pd.read_csv('./data/boston.csv')

# 正则化后的回归模型
def data_dispose(data_set):
    """
    desc:  数据处理  利用pandas库进行处理 返回numpy对象
    parameters:
        data_set  pandas类型  数据集
    return
        X_dispose np.array （m, n）  处理后的特征量
        Y_dispose np.array  (m,1)    处理后的真实值
        data_labels   list    (,n)      特征标签

    """
    data_set = pd.DataFrame(data_set)

    # 提取特征量、特征标签、真实值
    X_dispose = data_set.iloc[:, :-1]  # 特征量 ：除最后一列外都为特征
    Y_dispose = data_set.iloc[:, -1]  # 真实值：最后一列为真实值
    data_labels = data_set.columns  # 特征标签

    return np.array(X_dispose), np.array(Y_dispose, ndmin=2).reshape(-1, 1, ), data_labels
def Z_score_normalization(X):
    """
    desc:Z-score归一化
        公式：x* = ( x − μ ) / σ
    paremeters:
        X   np.array (m,n) 原始数据
    return:
        X_nor np.array (m,n)  归一化后的
    """
    # 计算样本的特征的均值和标准差
    Mu =    np.mean(X, axis=0)
    Sigma = np.std(X,  axis=0)

    # print(f"Mu = {Mu}")
    X_nor = (X - Mu) / Sigma

    return X_nor, Mu, Sigma
# 数据预处理:
X, Y, X_labels =  data_dispose(data_set)
# 数据归一化
X , mu , sigma =Z_score_normalization(X)




# 实例化一个神经网络模型
ann = AnnModel()
# 构建一个神经网络
ann.Sequential([
    # 隐藏层1
    Dense(neuron_num=15, active_fun="relu"),
    # Dense(neuron_num=10, active_fun="relu"),
    # 输出层
    Dense(neuron_num=1, active_fun="linear")

])

# 编译模型
lr=0.001
_lambda=0.1
epochs = 1000
validation_split=0.2


# 数据集分割
train_set, test_set = ann.data_set_splite((X, Y), validation_split=validation_split)

ann.compiler(
    lossType="mean_squared_error",
    gradientAlgorithm="mini_batch_gradient_descent",
    miniBatchSize=20,
    validation_splite=0.3,
    patience=100,
    lr=lr,
    _lambda=_lambda)


# 模型训练
W_opt, B_opt, min_cost = ann.fit(train_set, epochs=epochs)

print("traing result:")
table_fit = ColorTable(theme=Themes.OCEAN, field_names=['second_by_epoch', 'lr','iteration','regularized paremeter','feature count', 'train_data percent','min cost', 'opt_W', 'opt_b'])
table_fit.add_row([ann.second_enmu_iteration, lr,   epochs,  _lambda,   train_set['X'].shape[1],     str((1-validation_split)*100) + "%" , f'cost={round(min_cost, 5)}'  ,  '---',   '---'])
print(table_fit)

# 模型评价
MAE,RMSE,predict,Y= ann.evaluate(test_data=test_set)
print("model evaluate:")
table_evaluate = PrettyTable(['RMSE','MAE'])
table_evaluate.add_row([ round(RMSE, 4),MAE])
print(table_evaluate)



# 模型预测
predict_set={
    "X": np.array(train_set['X'][:10, :]),
    "Y": train_set['Y'][:10]
}
# 模型预测
predict_y,predict_label = ann.predict(predict_set)
print(f"Y = { predict_set['Y']}")
print(f"model predict result: {predict_y}")

# 用户自定义函数
def customize(net):
    """
    desc: 用户自定义函数
    paremeter： ann为该实例对象
    """
    # 绘制神经网络图
    # 实例化netGraph对象
    networkGraph = netGraph(type=1)
    input_layer = []

    # 增加输入层
    for n in  range(len(X_labels)):
        # 设置节点在网络中的位置
        pos = (1, n+1)
        # name名
        name = "$x_{%d}$" % (n+1)
        input_layer.append({
            "name": name,
            "pos":pos
        })
        # label设置
        label = X_labels[n]

        # 设置节点
        networkGraph.addNode(
            name=name,
            pos=pos,
            label=label
        )


    # 增加隐藏层和输出层
    for layer_n in range(len(net.layers)):
        """
        desc:遍历神经网络的结构层
        """
        # 遍历该层中的节点
        for n in range(len(net.layers[layer_n])):
            # 设置节点在网络中的位置
            pos = (layer_n+2, n + 1)
            # name名
            name = "$a^{(%d)}_{%d}$" % (layer_n + 2, n+1)
            # label设置
            # label = X_labels[n]
            # 增加网络节点
            networkGraph.addNode(
                name=name,
                pos=pos,
                # label="该节点node的描述label",
                # label_color="label的颜色，默认black",
                # 出度edge边信息
                # nexts=[
                #     {
                #         "node": "连接node的name",
                #         "label": "edge边标签",
                #         "color": "edge边标签颜色",
                #         "weight": edge边的权重
                #     },
                # ],
                previous=
                [
                    {
                        "node": node['name'],
                        "label": "",
                        "color": "blue",
                        "weight": 1
                    }
                    for node in input_layer
                ]
                if layer_n == 0
                else
                [
                    # 其它层
                    {
                        "node":"$a^{(%d)}_{%d}$" % (layer_n + 1, n+1),
                        # "label": "",
                        "color": "red",
                        "weight": 1
                    }
                    for n in range(len(net.layers[layer_n-1]))
                ]
            )
    # 绘制网络图
    networkGraph.draw()


ann.customize(customize)
