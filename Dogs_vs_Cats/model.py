import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, num_class):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 124)
        self.fc3 = nn.Linear(124, 64)
        self.fc4 = nn.Linear(64, num_class)
    
    def forward(self, xb):
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        xb = F.relu(self.fc3(xb))
        xb = self.fc4(xb)
        out = nn.Softmax(dim=1)(xb)
        return xb