import torch
from torch.utils import data
from d2l import torch as d2l

train, iter = d2l.evaluate_accuracy()
def synthetic_data(w, b, num_examples):
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return X, y.reshape(len(X), 1)


w_true = torch.Tensor([2, -3.4])
b_true = torch.Tensor([4.2])
features, labels = synthetic_data(w_true, b_true, 1000)
# 实例化TensorDataset类
# Dataset wrapping tensors.
# Each sample will be retrieved by indexing tensors along the first dimension.
# 接受变量并整合
dataset = data.TensorDataset(features, labels)
dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True, drop_last=False)

net = torch.nn.Sequential(torch.nn.Linear(2, 1))
# 初始化网络数值
net[0].weight.data.normal_(mean=0, std=0.01)
net[0].bias.data.fill_(0)

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

num_epochs = 3
for epoch in range(3):
    for i, data in enumerate(dataloader, 0):
        inputs, targets = data
        y_pred = net(inputs)
        l = loss(y_pred,targets)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    loss_train = loss(net(features), labels)
    print('epoch= ', epoch+1, '  loss=', loss_train.mean())