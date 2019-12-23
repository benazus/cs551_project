import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from preprocessing import PatternDataset

root_dir = "/home/bb/BilkentUniversity/Fall2019/cs551/project/dataset/"
data_dir = root_dir + "files/"
lr = 1e-3
batch_size = 32
momentum = 0.9

class CNN_MLP_ChannelConcat_Model(nn.Module):
    def __init__(self):
        super(CNN_MLP_ChannelConcat_Model, self).__init__()        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(2560, 128)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2,2))
        x = F.dropout(x, training=True)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2,2))
        x = F.dropout(x, training=True)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2,2))
        x = F.dropout(x, training=True)
        x = x.view(-1, 2560)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = F.relu(self.fc2(x))
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("---------Device: " + str(device))

dataset = PatternDataset(data_dir, root_dir + "labels.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
print("---------Train data ready.")

model = CNN_MLP_ChannelConcat_Model()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
epochs = 10
print("---------Training starting.")

# training
for i in range(epochs):
    running_loss = 0
    print("---------Epoch " + str(i))
    for j, data in enumerate(dataloader):
        rgb, labels = data[0].to(device), data[3].to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # stats
        running_loss += loss.item()
        if j % 50 == 49:
            print('[%d, %5d] loss: %.3f' % (i + 1, j + 1, running_loss / 10))
            running_loss = 0
print("---------Training complete.")
torch.save(model.state_dict(), "CNN_MLP_ChannelConcat_Model.pth")