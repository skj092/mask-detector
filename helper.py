# import cv2 
# import matplotlib.pyplot as plt
# import cv2 

# # visualize the image and bounding box
# def show_img_bbox(img, target):
#     fig = plt.figure(figsize=(10,10))
#     bbox = target['boxes']
#     label = target['labels']
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     for i in range(len(bbox)):
#         xmin = bbox[i][0]
#         ymin = bbox[i][1]
#         xmax = bbox[i][2]
#         ymax = bbox[i][3]
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#         cv2.putText(img, str(label[i]), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     plt.imshow(img)
#     plt.show()
    
# # label2 class
# def class2lbl(label):
#     if class = 1:
#         return 'mask'
#     else:
#         return 'no_mask'