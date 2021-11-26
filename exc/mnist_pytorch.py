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
                                                num_workers=0)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.dense = torch.nn.Sequential(torch.nn.Linear(28 * 28, 10))

    def forward(self, x):
        x = self.dense(x)
        return x


model = Model()
model = model.cuda()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 2
# model.load_state_dict(torch.load('model_parameter.pkl'))
running_loss = 0.0
running_correct = 0
for epoch in range(n_epochs):
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    iterator = tqdm(data_loader_train)
    for data in iterator:
        running_loss = 0.0
        X_train, y_train = data
        X_train = X_train.cuda()
        y_train = y_train.cuda()

        X_train = X_train.view(-1, 28 * 28)
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
        # running_correct += torch.sum(pred == y_train.data)
    # testing_correct = 0
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