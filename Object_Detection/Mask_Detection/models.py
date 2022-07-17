import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys 
from dataset import MaskDetectionDataset, get_transform
from torch.utils.data import DataLoader
sys.path.append('../')
from vision.references.detection import utils


def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

if __name__=="__main__": 
    # testing the model 
    model = get_object_detection_model(2)
    train_dataset = MaskDetectionDataset('train', transforms= get_transform(train=True))
    train_dl = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    # testing a batch of images
    # For Training
    images,targets = next(iter(train_dl))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    predictions = model(images, targets)
    print(predictions)
