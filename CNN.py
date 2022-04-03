from msilib import type_key
import os
import io
import numpy as np
from skimage import io
# from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.functional as F
# import torch.optim as optim
#加载数据
def loadData(filePath, sampleSize, pixelSize, channels):
    X = np.zeros((sampleSize, pixelSize, pixelSize, channels))
    Y = np.zeros((sampleSize, 1))
    image = os.listdir(filePath)
    i = 0
    for item in image:
        # print(type(item))
        #break
        img = io.imread(filePath+item)
        X[i]=img
        angle=item.split("_ang_")[1]
        angle=angle.split(".jpg")[0]
        Y[i]=float(angle)
        i=i+1
    X = np.swapaxes(X,1,3)
    return X,Y
#定义神经网络模型
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=5, padding=2)
        self.max_pool8 = torch.nn.MaxPool2d(8, 8)
        self.fc1 = nn.Linear(300, 1)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.max_pool8(out)
        out = out.view(out.size(0), -1)
        #print(out.size())
        #ps = raw_input()
        out = torch.sigmoid(self.fc1(out))
        return out
def main():
    #加载训练集
    X_train,Y_train = loadData(r'd:\tempworkspace\python_workspace\ship_heading_detection\train_data\\', 100, 300, 3)
    #加载测试集
    X_test,Y_test = loadData(r'd:\tempworkspace\python_workspace\ship_heading_detection\test_data\\', 100, 300, 3)
    #数据格式转化
    train = data_utils.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    test = data_utils.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
    train_loader = data_utils.DataLoader(train, batch_size=1, shuffle=True)
    test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=False)

    print("train:"+str(type(train))+" test:"+str(type(test))+" train_loader:"+str(type(train_loader))+" test_loader:"+str(type(test_loader)))
    #初始化神经网络
    net = MyCNN()
    criterion = nn.MSELoss()
    #构建随机梯度下降算法
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    #训练模型
    for epoch in range(1,10,1):
        net.train()
        tot_loss_train = 0.0
        for (inputs, targets) in train_loader:
            print(type(inputs))
            print(inputs[0])
            print(inputs[0][0])
            print(inputs[0][0][0])
            print(type(inputs[0][0][0][0]))
            print(str(inputs[0][0][0][0]))
            #break
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_loss_train += loss.item()
        net.eval()
        tot_loss_test = 0.0
        for (inputs, targets) in test_loader:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            tot_loss_test += loss.item()
        print(epoch, tot_loss_train, tot_loss_test)
    net.eval()
    #测试模型
    for (inputs, targets) in test_loader:
        outputs = net(inputs)
        print(outputs.data.numpy())
        print(targets.data.numpy())
        ps = raw_input()
main()

# test_x = Variable(torch.unsqueeze(test_data.test_data,dim  = 1),volatile = True).type(torch.FloatTensor)[:500]/255. #（0-1）
# test_y = test_data.test_labels[:500].numpy()
# print(X_test.shape)
# print(test_x.shape)
# print(test_y.shape)



