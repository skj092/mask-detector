
# AIM: To Build and Object Detection model which can able to detect whether person on the image wearing mask or not. 

![image](https://user-images.githubusercontent.com/43055935/182284500-d7335b2f-54e0-45da-a548-79de6c064cdd.png)
![image](https://user-images.githubusercontent.com/43055935/182284606-61f6b158-7f1c-43e8-87d4-7934df7dc950.png)

# Requirements:
1.  Dataset:
We will downlaod the dataset from roboflow. Roboflow hosts free public computer vision datasets in many popular formats (including CreateML JSON, COCO JSON, Pascal VOC XML, YOLO v3, and Tensorflow TFRecords).

        `Dataset contain 3 folder - train, valid and test and every files in the folder are in the format imagename.png and imagename.xml.` 

        `There is an incorrect labeled file `0_8w7mkX-PHcfMM5s6_jpeg.rf.039eb72a4757882968537a6ae94d198f.xml` which we have to remove.`

2. We require to clone or import the following files from torchvision library:
* coco_eval.py
* coco_utils.py
* transforms.py
* utils.py 
* engine.py 

# Algorithm:

We will use `Faster RCNN` for the task. 

![image](https://user-images.githubusercontent.com/43055935/182286323-6273e363-2dac-4cc4-ab14-099b5b50f40c.png)


## Faster RCNN have 4 main sections:

1. Backbone CNN - It contain a pretrained CNN network mostly VGG or Resnet. The goal of this network is to extract the feature map from the image. 
2. Region Proposal Network: It contain CNN to propost regions of interest (RoI) from the feature map and product 2 outputs; the "objectness score"(object or no object) and the box locations.
3. ROI Pooling: It make all the boxes from RPN ourput to same size.
4. Head or Output Layer: It contain two fully-connected layers: 
    1. A softmax classifier to output the class probability, and 
    2. A bounding-box regressin CNN to the bounding box predictions.

# Procedure:

1. Install and import all the necessary packages. 
2. Make a pytorch custom dataset and dataloader for the Object Detection task. The dataset must product 2 outputs. 
    1. images - A tensor datatype
    2. target - A dict type contain bounding box co-ordinates, labels, image_id, area, and iscrowd. 
3. Define the model and customize it to fit for out task. 
4. Set all the hyperparameters and train the network using functions from torchvision engine.py file.


# Observations:

We have trained the model for 10 epochs which took half and hour. 

* epochs = 10
* total train loss = 0.3024
* total valid loss = 0.4173
* train classification loss = 0.1381
* valid classification loss = 0.1774
* train regression loss = 0.1149
* valid regression loss = 0.1929

# Conclusion: 
Model is overfitting, there are few things we could to. We haven't used data augmentation so that is the first step. 

# Result: 
For such an small dataset the score looks fine. And we are able to build successuflly an Object Detection model using pytorch. 
