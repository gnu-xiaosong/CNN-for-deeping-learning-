import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from package_xskj_NetworkXsimple import netGraph
from pylab import mpl
import plotly.express as px

# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
sns.set_theme()
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体

# 实例化netGraph对象
networkGraph =netGraph(type=1)




def plot_data_scatter(data_set):
    """
    desc：绘制原始数据的散点图
    paremeters:
        data_set pandas 数据集
    """
    fig = px.scatter(data_set, x=str(data_set.columns[0]), y=str(data_set.columns[-1]))
    # 打印输出

    return fig



def plot_feature_weight(feature_labels, opt_W, opt_b):
    """
    desc：绘制模型特征权重信息
    paremeters:
        net       netGraph   netGraph对象
        feature_labels  list 特征名称
        opt_W     list  最优模型参数W
        opt_b     float 最优模型偏置b
    """
    # 转为数组list
    opt_W = opt_W.reshape(-1)


    for i in range(len(opt_W)):
        # 节点坐标(层，节点数)
        pos = (1, i + 1)
        # 节点名称
        name = f"x{i + 1}"
        # 节点标签
        label = feature_labels[i]
        # print(f"feature_labels={label}")
        # 增加网络节点
        networkGraph.addNode(
            name=name,
            pos=pos,
            label=label,
            label_color="red",
            nexts=[
                {
                    "node": "f(X)",
                    "label": f"w_{ i +1}",
                    "color": "blue",  # 颜色
                    "weight": round(opt_b, 2)  # edge权重
                }
            ]
        )

    # 偏置节点
    networkGraph.addNode(
        name="b",
        pos=(1, len(opt_W) + 1),
        label="",
        label_color="red",
        nexts=[
            {
                "node": "f(X)",  # 连接节点
                "label": "b",  # edge标签
                "color": "blue",           # 颜色
                "weight":  round(opt_b, 2)  # edge权重
            }
        ]
    )


def plot_feature_y(X ,X_label ,Y):
    """
    desc: 特征量与真实值的相关性
    """
    m ,n = X.shape

    axs = []
    # 设置画布
    fig = plt.figure(figsize=(14 ,14), dpi=100)
    plt.subplots_adjust(bottom=0, right=0.8, top=1, hspace=0.5)
    # 列
    coloum = 3
    for i in range(n):
        ax = fig.add_subplot(math.ceil( n /coloum) , coloum, i+ 1)
        if i == 0:
            ax.set_ylabel('真实值 y')
        ax.set_xlabel('x')
        ax.set_title(X_label[i])

        # 绘制散点图
        ax.scatter(X[:, i], Y)
        # 绘制箱型图
        np.random.seed(10)  # 设置种子
        D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))
        ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "white", "linewidth": 0.5},
                   boxprops={"facecolor": "C0", "edgecolor": "white",
                             "linewidth": 0.5},
                   whiskerprops={"color": "C0", "linewidth": 1.5},
                   capprops={"color": "C0", "linewidth": 1.5})

        axs.append(ax)


def plot_cost(cost):
    """
    desc:绘制损失值图
    """
    fig, ax = plt.subplots()
    ax.set_title("代价变化图")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    plt.plot(cost)
    plt.show()

def plot_corr(data_set):
    """
    desc:绘制变量间的相关性关系
    """
    # 计算变量间的相关系数
    corr = data_set.corr()

    f, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("变量之间的相关性系数值")
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, cmap="YlGn", ax=ax)
