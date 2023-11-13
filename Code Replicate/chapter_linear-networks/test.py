import matplotlib.pyplot as plt
import torch
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device}')
def data_generate(w, b, num_samples):
    X = torch.normal(mean=0, std=1, size=(num_samples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return X, y.reshape(len(X), 1)


w = torch.normal(mean=0, std=0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(w)
print(b)

true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features, labels = data_generate(true_w, true_b, 1000)

plt.figure(figsize=(4,3))
# features是一个mxn矩阵(m:样本容量，n:特征) labels是对应目标标签， 1为散点大小
# 第一个特征和和labels的散点图
# plt.scatter(features[:, 0], labels, 1)
# 第二个特征和labels的散点图
plt.scatter(features[:, 1], labels, 1)
plt.show()


def dataloader(batch_size, features, labels):
    data_amt = len(features)
    data_indices = list(range(data_amt))
    random.shuffle(data_indices)
    for i in range(0, data_amt, batch_size):
        batch_indices = torch.tensor(data_indices[i: min(i+batch_size, data_amt)])
        yield features[batch_indices], labels[batch_indices]


def linear_reg(X, w, b):
    return torch.matmul(X, w) + b


def square_loss(y_pred, y_true):
    return (y_pred - y_true)**2 / 2


def sqd(parameters, lr, data_amt):
    with torch.no_grad():
        for parameter in parameters:
            parameter -= lr * parameter.grad / batch_size
            parameter.grad.zero_()


num_epoch = 3
lr = 0.001
batch_size = 16

for epoch in range(num_epoch):
    for X, y in dataloader(batch_size, features, labels):
        l = square_loss(linear_reg(X, w, b), y)
        batch_amount = len(l)
        print(batch_amount)
        l.sum().backward()
        sqd([w, b], lr, batch_amount)
    with torch.no_grad():
        train_l = square_loss(linear_reg(features, w, b), labels)
        print('epoch=', epoch+1, '  loss={:.8f}'.format(train_l.mean()))

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
