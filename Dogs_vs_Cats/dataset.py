from torch.utils.data import Dataset, DataLoader
import os 
from glob import glob 
from torchvision import transforms, utils
import random 
from PIL import Image
from pathlib import Path 
from config import tfms
import torch 




class DogsVsCatsDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = tfms
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).resize((224,224))
        label = img_path.name.split('.')[0]
        label = 1 if label == 'cat' else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

if __name__ == '__main__':
    train_dataset = DogsVsCatsDataset(train_images)
    valid_dataset = DogsVsCatsDataset(valid_images)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    print(len(train_loader))
    print(len(valid_loader))