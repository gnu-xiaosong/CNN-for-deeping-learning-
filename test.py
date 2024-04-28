import pickle

import numpy as np





from Conv import Conv, Pooling

data = np.array([
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ],
    [
        [1, 1, 1,  1],
        [1, 1, 1,  1],
        [1, 1, 1,  1],
    ]
])
filter_data = np.array([
    [
        [1, -1],
        [1, -1]
    ],
    [
        [1, 1],
        [-1, -1]
    ]
])

filter_ = []

filter_.append(filter_data)
filter_.append( np.array([
    [
        [1, -1],
        [1, 0]
    ],
    [
        [0, 1],
        [-1, -1]
    ]
]))
filter_.append(filter_data)
filter_.append(np.array([
    [
        [3, -3],
        [3, 0]
    ],
    [
        [0, 2],
        [-2, -1]
    ]
]))
# 实例化卷积层
conv = Conv()
data1 = conv.conv_operate(data, filter_)

print("----------------convalution ------------------")
print(data1)

# 池化操作
# 实例化
pooling = Pooling(f=2, s=1)
pooling_data = pooling.pooling_operate(data)
print("-------pooling data------")
print(f"{pooling_data}")

exit()



class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            # 每次迭代时执行特定的操作
            print("Performing specific operation for iteration", self.index + 1)
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            # 迭代结束时抛出 StopIteration 异常
            raise StopIteration

# 创建自定义迭代器的实例
my_iterator = MyIterator([1, 2, 3, 4, 5])

# 使用迭代器进行迭代
for item in my_iterator:
    print("Iteration result:", item)





def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# read file content

file_name = "./data/cifar-10-batches-py/train/data_batch_1"
data =dict(unpickle(file_name))

batch_name = data[b"batch_label"]
X =       data[b"data"]     # 特征数据
Y =       data[b"labels"]   # 真实值：对应索引
X_path = data[b"filenames"] # 文件路径

print(data)
print(f"X shape ={X.shape} Y shape={np.array(Y).shape} X_path shape={np.array(X_path).shape}")