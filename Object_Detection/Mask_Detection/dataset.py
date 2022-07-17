import os 
from pathlib import Path 
import xml.etree.ElementTree as ET 
from torch.utils.data import Dataset
from PIL import Image
import torch 
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2 
import sys 
sys.path.append('../')
from vision.references.detection import utils

# def transform
# # def transform
# def get_transform(train=True):
#     if train:
#         return A.Compose([
#             A.RandomCrop(height=300, width=300, p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomBrightness(p=0.5),
#             A.RandomContrast(p=0.5),
#             A.RandomGamma(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.PadIfNeeded(min_height=512, min_width=512, p=1),
#             A.Normalize(p=1, max_pixel_value=255.0, always_apply=True),
#             ToTensorV2(p=1.0)
#         ] , bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
#     else:
#         return A.Compose([
#             A.PadIfNeeded(min_height=300, min_width=512, p=1),
#             A.Normalize(p=1, max_pixel_value=255.0, always_apply=True),
#             ToTensorV2(p=1.0)
#         ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class MaskDetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.image_paths = list(self.root.glob('*.jpg'))
        # removing an exception
#         self.image_paths.remove("train/0_8w7mkX-PHcfMM5s6_jpeg.rf.039eb72a4757882968537a6ae94d198f.jpg")
        self.image_paths.sort()
        self.mask_paths = [p.parent / (p.stem + '.xml') for p in self.image_paths]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        # image = cv2.imread(str(image_path))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(str(image_path)).convert('RGB')
        
        # Parsing xml to get the labels and the bounding boxes
        annotation = ET.parse(mask_path)
        root = annotation.getroot()
        objects = root.findall('object')
        boxes = []
        labels = []
        for obj in objects:
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)

        # Converting the labels and the bounding boxes to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = list(map(lambda x: 1 if x == 'mask' else 0, labels))
        labels = torch.as_tensor(labels)
        image_id = torch.tensor([idx])

        # Other parameters
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Creating the tensor for the dataset
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        # Applying the transforms
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


    def __len__(self):
        return len(self.image_paths)
            
if __name__=="__main__":
    ds = MaskDetectionDataset(root='train/', transforms=get_transform(train=True))
    for i in range(len(ds)):
        image, target = ds[i]
        print(target['image_id'], target['boxes'], target['labels'])