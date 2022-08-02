import torch 
from models import get_object_detection_model
from dataset import MaskDetectionDataset, get_transform
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch.utils.data import Subset 


if __name__=="__main__":
    train_ds = MaskDetectionDataset(root='train/', transforms=get_transform(train=True))
    valid_ds = MaskDetectionDataset(root='valid/', transforms=get_transform(train=False))

    train_ds = Subset(train_ds, range(0, len(train_ds)//3, 2))
    valid_ds = Subset(valid_ds, range(0, len(valid_ds)//3, 2))

    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)


    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    num_classes = 2

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    # training for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, train_dl, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_dl, device=device)
                        