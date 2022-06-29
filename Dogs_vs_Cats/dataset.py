from torch.utils.data import Dataset, DataLoader
import os 
from glob import glob 
from torchvision import transforms, utils
import random 
from PIL import Image
from pathlib import Path 
from 

p = Path('.')
images = list(p.glob('input/train/*.jpg'))
random.shuffle(images)
train_images, valid_images = images[:int(len(images) * 0.8)], images[int(len(images) * 0.8):]



class DogsVsCatsDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = img_path.name.split('.')[0]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    train_dataset = DogsVsCatsDataset(train_images)
    print(train_dataset[0])