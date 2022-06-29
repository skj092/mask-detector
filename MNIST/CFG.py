from torchvision import transforms
import torch 

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()
])