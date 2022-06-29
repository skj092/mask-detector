from torchvision import transforms 

tfms = transforms.Compose([transform.Resize((224, 224), transform.ToTensor())])