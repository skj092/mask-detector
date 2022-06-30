from model import CNN_NET, vgg
from dataset import DogsVsCatsDataset
import config
import torch

if __name__=="__main__":
    train_data = DogsVsCatsDataset(config.train_images)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    xb, yb = next(iter(train_loader))
    print(xb.shape, yb.shape)
    net = vgg()
    print(net)
    print(net(xb).shape)