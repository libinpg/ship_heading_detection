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
X_train = np.zeros((1748, 300, 300, 3))
Y_train = np.zeros((1748, 1))

filepath_train=r'd:\tempworkspace\python_workspace\ship_heading_detection\train_data\\'
train_image= os.listdir(filepath_train)
# print(type(train_image))
i = 0
for item in train_image:
    # print(type(item))
    #break
 
    image_train = io.imread(filepath_train+item)
    X_train[i]=image_train
    angle_train=item.split("_ang_")[1]
    angle_train=angle_train.split(".jpg")[0]
    Y_train[i]=float(angle_train)
    i=i+1
X_train = np.swapaxes(X_train,1,3)



X_test = np.zeros((1228, 300, 300, 3))
Y_test = np.zeros((1228, 1))

filepath_test=r'd:\tempworkspace\python_workspace\ship_heading_detection\test_data\\'
test_image= os.listdir(filepath_test)
# print(type(train_image))
j = 0
for item in test_image:
    # print(type(item))
    #break
 
    image_test = io.imread(filepath_test+item)
    X_test[j]=image_test
    angle_test=item.split("_ang_")[1]
    angle_test=angle_test.split(".jpg")[0]
    Y_test[j]=float(angle_test)
    j=j+1
X_test = np.swapaxes(X_test,1,3)

















train = data_utils.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
test = data_utils.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
train_loader = data_utils.DataLoader(train, batch_size=1, shuffle=True)
test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=False)



print("train:"+str(type(train))+" test:"+str(type(test))+" train_loader:"+str(type(train_loader))+" test_loader:"+str(type(test_loader)))





net = MyCNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

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

for (inputs, targets) in test_loader:
    outputs = net(inputs)
    print(outputs.data.numpy())
    print(targets.data.numpy())
    ps = raw_input()


# test_x = Variable(torch.unsqueeze(test_data.test_data,dim  = 1),volatile = True).type(torch.FloatTensor)[:500]/255. #（0-1）
# test_y = test_data.test_labels[:500].numpy()
# print(X_test.shape)
# print(test_x.shape)
# print(test_y.shape)



