import matplotlib.pyplot as plt
import torch
import random


# 生成y=Xw+b+噪声
def synthetic_data(w, b, num_examples):
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))
    # 生成一个大小为 num_examples(样本容量:样本个数)*len(w)(特征个数,与权重个数相同)
    # 例如每个样本有两个特征，一共有一千个样本，则构成矩阵为 𝐗∈ℝ1000×2 w∈ℝ2×1
    # X*w∈ℝ1000×1为1000个预测输出
    y = torch.matmul(X, w) + b  # matrix multiplication 矩阵积，非叉乘
    # add noise
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return X, y.reshape((-1, 1))
    # X.shape torch.Size([1000, 2])  二维张量/矩阵
    # y.shape torch.Size([1000, 1])  一维张量/向量
    # 在机器学习中，尤其是在使用 PyTorch 等深度学习框架时，通常对标签进行二维张量的 reshape 是为了保持一致性。
    # 这可以帮助避免在计算中引入不必要的复杂性，并使代码更加清晰。为了保持一致性，可以将标签 y reshape成一个形状为 1000×1的列向量
    # 这样它的形状就与模型输出的形状一致了。这在许多深度学习框架中是一个通用的做法，它有助于简化代码，并确保在处理模型输出和标签时没有维度不匹配的问题


# torch.Tensor() 和 torch.tensor()的区别
# torch.Tensor() 是一个构造函数，用于创建张量，可以用于创建空张量或从现有张量创建新张量。
# torch.tensor() 是一个工厂函数，用于从给定数据创建新的张量，并且能够直接从 Python 列表、NumPy 数组等数据类型中创建。
true_w = torch.Tensor([2, -3.4])
true_b = torch.Tensor([4.2])
features, labels = synthetic_data(true_w, true_b, 1000)

plt.figure(figsize=(4,3))
# features是一个mxn矩阵(m:样本容量，n:特征) labels是对应目标标签， 1为散点大小
# 第一个特征和和labels的散点图
# plt.scatter(features[:, 0], labels, 1)
# 第二个特征和labels的散点图
plt.scatter(features[:, 1], labels, 1)
plt.show()


# 从原始数据集中获取小批量数据
def data_iter(batch_size, features, labels):
    # len()会选取第一维的长度 = 1000
    nums_features = len(features)
    # 利用这个长度生成一个列表包含同等数量的元素为0-nums_features, 并打乱
    indices = list(range(nums_features))
    random.shuffle(indices)
    # 从0开始直到最后一个元素,每次抽取一个batch_size
    for i in range(0, nums_features, batch_size):
        # 定义一个张量,每次从随机的下标列表中从i元素开始抽取一个batch,除非最后一个batch,抽到最后一个元素位置(如果剩余元素少于batch_size)
        batch_indicices = torch.tensor(indices[i : min(i+batch_size, nums_features)])
        # 使用 yield 关键字生成一个批量的特征和对应的标签，并返回给调用方。
        # yield 的作用是将函数变成一个生成器，允许你在迭代中逐步产生值，而不是一次性生成所有值。
        # 定义一个循环，可以循环接受生成器中的返回值，循环条件每执行一次，就会抽取一次数据
        # 相当于一个dataloader
        # !!直接使用列表也可以
        yield features[batch_indicices], labels[batch_indicices]
        # 如果batch_size=10就会返回10组数据，并且等待下一次抽取(也就是下一次执行循环条件)
        """
        batch_size = 10
        batch_index = 0
        for X, y in data_iter(batch_size, features, labels): 抽取数据，这就是yield和return的区别
            # return会一次性返回所有数据
            # yield会将函数转变为一个生成器，允许在迭代中逐步产生值，而不是一次性产生值
            print(X, '\n', y)
            print(batch_index)
            batch_index += 1
        """


batch_size = 10
# 初始化权重与偏置，两者都是我们要更新的参数，所以设置保留梯度
# 权重一般初始化为mean=0, std=0.01的正态分布
# 偏执一般设置为0
w = torch.normal(mean=0, std=0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linear_reg(X, w, b):
    # 构建线性化模型与创建数据集的模型相同，只是数据不同
    # (1000,2)*(2,1) => (1000,1)
    return torch.matmul(X, w) + b


def squared_loss(pred, true):
    # pred是从模型中生成的 维度为(1000,1)
    # true同样也是模型生成的 维度为(1000,1)
    # 个人人为不需要添加reshape, 代码确实可以跑通
    # return (pred - true.reshape(pred.shape)) ** 2 / 2
    return (pred - true)**2 / 2


# 优化器, 优化器中的梯度更新为数字运算，不需要计算梯度
# parameters传入的是所有需要更新的参数
def sgd(parameters, learning_rate, batch_length):
    with torch.no_grad():
        for parameter in parameters:
            # 这里的batchsize 其实有问题
            # 抽取的indicices并不一定是batch_size
            # 查看打印结果会发现最后一个batch只有8个样本
            # 为什么要除以样本容量？因为最后是对loss.sum()求梯度
            parameter -= learning_rate * parameter.grad / batch_length
            # 清空梯度
            parameter.grad.zero_()


lr = 0.001
num_epoch = 3
net = linear_reg
loss = squared_loss

for epoch in range(num_epoch):
    # 抽取一组minibatch的样本(输入与输出对应关系)
    for X, y in data_iter(batch_size=batch_size, features=features, labels=labels):
        # 计算损失, y_hat是数据经过当前网络得到的输出
        l = loss(net(X, w, b), y)
        batch_length = len(l)
        # 对loss求和, 并计算bp
        l.sum().backward()
        # 对目标变量更新
        sgd([w ,b], lr, batch_length)
    with torch.no_grad():
        # train_l.shape = torch.Size([1000, 1])
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

