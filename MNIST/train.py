from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torch
import numpy as np 
from model import Net
import CFG
import torch.nn as nn 
from engine import train
from CFG import transform

if __name__=="__main__":
    image_path = '.'
    mnist_train_ds = MNIST(root=image_path, train=True, transform=transform, download=True)
    mnist_test_ds = MNIST(root=image_path, train=False, transform=transform, download=True)
    train_ds, valid_ds = torch.utils.data.random_split(mnist_train_ds, [50000,10000], torch.Generator().manual_seed(42))

    train_ds = Subset(train_ds, np.arange(1024))
    valid_ds = Subset(valid_ds, np.arange(512))

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.BATCH_SIZE)

    model = Net(input_size=784, num_class = 10)
    model.to(CFG.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    device = CFG.device
    
    train(model, train_loader,valid_loader, optimizer=optimizer, loss_fn=loss_fn)
    torch.save(model.state_dict(), './model.pt')