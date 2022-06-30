from dataset import DogsVsCatsDataset
from torch.utils.data import DataLoader 
from model import Net, CNN_NET, vgg
from engine import train
import config 
import torch 
import torch.nn as nn 
import numpy as np 
from torch.utils.data import Subset

if __name__=="__main__":
    train_dataset = DogsVsCatsDataset(config.train_images)
    valid_dataset = DogsVsCatsDataset(config.valid_images)
    train_dataset = Subset(train_dataset, np.arange(1024))
    valid_dataset = Subset(valid_dataset, np.arange(512))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE)
    # model = Net(input_size=config.INPUT_SIZE, num_class = config.NUM_CLASS)
    # model = CNN_NET()
    model = vgg()
    model.to(config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    device = config.device
    train(model, train_loader,valid_loader, optimizer=optimizer, loss_fn=loss_fn)