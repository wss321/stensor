import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor()])
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

batch_size = 64
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=8)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=8)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.l1 = torch.nn.Linear(8 * 28 * 28 // 16, 64)
        self.l2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu3(x)
        x = self.l2(x)

        return x


model = Model()
model = model.cuda()
torch.manual_seed(1234)
# torch.cuda.manual_seed(1234)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
n_epochs = 10
# model.load_state_dict(torch.load('model_parameter.pkl'))
running_loss = 0.0
from time import time

data_list = []
for data in data_loader_train:
    running_loss = 0.0
    X_train, y_train = data
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    data_list.append([X_train, y_train])
start = time()
for epoch in range(n_epochs):
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    running_correct = 0
    iterator = tqdm(data_list)
    for data in iterator:
        running_loss = 0.0
        X_train, y_train = data
        # X_train = X_train.cuda()
        # y_train = y_train.cuda()

        X_train = X_train.view(-1, 1, 28, 28)
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iterator.set_description("loss:{}".format(running_loss))
        # print(loss.item())
        running_correct += torch.sum(pred == y_train)
    print("train acc:", running_correct.item() / len(data_train))
    testing_correct = 0
    # for data in data_loader_test:
    #     X_test, y_test = data
    #     X_test, y_test = Variable(X_test), Variable(y_test)
    #     outputs = model(X_test)
    #     _, pred = torch.max(outputs.data, 1)
    #     testing_correct += torch.sum(pred == y_test.data)
    # print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
    #                                                                                   100 * running_correct / len(
    #                                                                                       data_train),
    #                                                                                   100 * testing_correct / len(
    #                                                                                       data_test)))
# torch.save(model.state_dict(), "model_parameter.pkl")
print("Time cost:{} s".format(time() - start))
