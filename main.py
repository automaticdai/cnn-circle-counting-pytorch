import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

import models

# parameters for train
train_on = False
train_data_path = "./data/img_data.csv"
model_save_dir_temp = "./models/temp/"
model_save_path = "./models/my_model.mdl"
Save_tmp_model = False
#net = models.AlexNet()
#net = models.Vgg16()
net = models.ResNet18()
train_batch_size = 1
train_epoch = 100
print_iter = 20


# parameters for test
test_on = True
test_data_path = "./data/img_data_test_p.csv"
model_load_path = "./models/my_model.mdl"


# Dataset Loader
class CircleDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (H, W, C)
        # can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((64, 64, 1))
        label = np.asarray(self.data.iloc[index, 0], dtype=np.float32).reshape((1))
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# train
def training(device):
    # load training data
    trans = transforms.Compose([transforms.ToTensor()])
    trainset = CircleDataset(train_data_path, transform=trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    # load network model
    net.to(device)
    print(net)

    # criterion = nn.CrossEntropyLoss() # a Classification Cross-Entropy loss
    criterion = nn.MSELoss().to(device)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adamax(net.parameters())

    # train the network
    for epoch in range(train_epoch):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get one input from the training set
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # display data
            # print((labels, inputs))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # print every # of mini-batches
            running_loss += loss.item()
            if i % print_iter == print_iter - 1:
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / print_iter))
                running_loss = 0.0


        if Save_tmp_model is True:
            # one epoch finished, save temp model
            t_save_path = model_save_dir_temp + "model_temp_{0:d}.mdl".format(epoch)
            torch.save(net, t_save_path)

    # all epochs are finished
    print('Training is finished')
    torch.save(net, model_save_path)


# test
def testing(device):
    # load test set
    trans = transforms.Compose([transforms.ToTensor()])

    testset = CircleDataset(test_data_path, transform=trans)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    # load model
    model = torch.load(model_load_path)
    model.to(device)

    # print model summary
    summary(model, input_size=(1, 64, 64))

    # statistics
    tp = 0
    total = 0
    mae = 0
    mse = 0

    for i, data in enumerate(testloader, 0):
        # get one (input, label) pair from the training set
        inputs, labels = data
        # print(intputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)

        # feed input into the network
        outputs = model(inputs)

        print("{0:d} {1:d}".format(int(labels), round(float(outputs))))

        total = total + 1
        error = abs(float(outputs) - float(labels))
        mae = mae + error
        mse = mse + error * error
        if abs(int(labels) - round(float(outputs))) <= 0:
            tp = tp + 1

    print("{0:.3f}, {1:.3f}, {2:d}, {3:d}".format(mae/total, mse/total, tp, total))


# main
if __name__ == '__main__':
    # use GPU to train
    processor = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("Device {} is used".format(processor))

    # train
    if train_on:
        training(processor)

    # test
    if test_on:
        testing(processor)
