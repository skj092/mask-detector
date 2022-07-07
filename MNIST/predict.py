import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from CFG import transform
import CFG  
from model import Net

# loading the model
def predict(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(CFG.device)
            images = images.reshape(images.shape[0], -1)
            labels = labels.to(CFG.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total



if __name__=="__main__":
    test_ds = MNIST(root="MNIST/raw", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE)
    model = Net(input_size=784, num_class = 10)
    model.to(CFG.device)
    model.load_state_dict(torch.load('./model.pt'))
    accuracy = predict(model, test_loader)
    print("Accuracy: ", accuracy)
    
