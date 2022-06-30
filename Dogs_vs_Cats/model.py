import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, input_size, num_class):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 124)
        self.fc3 = nn.Linear(124, 64)
        self.fc4 = nn.Linear(64, num_class)
    
    def forward(self, xb):
        xb = xb.reshape(xb.shape[0], -1)
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        xb = F.relu(self.fc3(xb))
        xb = self.fc4(xb)
        out = nn.Softmax(dim=1)(xb)
        return xb


class CNN_NET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.linear = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.linear(x.view(x.shape[0], -1))
        x = nn.Softmax(dim=1)(x)
        return x

class vgg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = models.vgg16(pretrained=True)
        self.last_linear = nn.Linear(1000, 10)
    def forward(self, x):
        x = self.features(x)
        x = self.last_linear(x)
        x = nn.Softmax(dim=1)(x)
        return x

       