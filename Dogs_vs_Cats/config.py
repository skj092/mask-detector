from torchvision import transforms 
import torch 
from glob import glob 
from pathlib import Path
import random 

p = Path('.')
images = list(p.glob('input/train/*.jpg'))
random.shuffle(images)
train_images, valid_images = images[:int(len(images) * 0.8)], images[int(len(images) * 0.8):]


tfms = transforms.Compose([transforms.ToTensor()])

INPUT_SIZE = 224*224*3
NUM_CLASS = 2
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'