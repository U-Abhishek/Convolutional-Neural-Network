# Deep-Learning
## 1) Image Segmentation using U-NET
#### NOTEBOOK NAME: UNET_Implementation.ipynb
Building your own U-Net, a type of CNN designed for quick, precise image segmentation, and using it to predict a label for every single pixel in an image - in this case, an image from a self-driving car dataset. 

This type of image classification is called semantic image segmentation. It's similar to object detection in that both ask the question: "What objects are in this image and where in the image are those objects located?," but where object detection labels objects with bounding boxes that may include pixels that aren't part of the object, semantic image segmentation allows you to predict a precise mask for each object in the image by labeling each pixel in the image with its corresponding class. The word “semantic” here refers to what's being shown, so for example the “Car” class is indicated below by the dark blue mask, and "Person" is indicated with a red mask:
![carseg](https://user-images.githubusercontent.com/86155658/132116430-f98b7960-980e-4501-8eb8-4b2970cc55a5.png)\
As you might imagine, region-specific labeling is a pretty crucial consideration for self-driving cars, which require a pixel-perfect understanding of their environment so they can change lanes and avoid other cars, or any number of traffic obstacles that can put peoples' lives in danger.

#### This notebok consists of:
1. Build your own U-Net.
2. Explain the difference between a regular CNN and a U-net.
3. Implement semantic image segmentation on the CARLA self-driving car dataset.[https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge].
4. Apply sparse categorical crossentropy for pixelwise prediction.

## 2) Transfer Learning with MobileNetV2
#### NOTEBOOK NAME: Transfer_learning_with_MobileNet_v2
Be using transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier!
![image](https://user-images.githubusercontent.com/86155658/133078539-fe7ffc74-cab7-407b-b85d-deed6b0f64ed.png)

A pre-trained model is a network that's already been trained on a large dataset and saved, which allows you to use it to customize your own model cheaply and efficiently. The one you'll be using, MobileNetV2, was designed to provide fast and computationally efficient performance. It's been pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes.

1) Create a dataset from a directory
2) Preprocess and augment data using the Sequential API
3) Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
4) Fine-tune a classifier's final layers to improve accuracy 

## 3) Residual Networks
#### NOTEBOOK NAME: Residual_Networks
Building a very deep convolutional network, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allow you to train much deeper networks than were previously feasible.
Building a ResNet50 model to classify hand signs
![image](https://user-images.githubusercontent.com/86155658/133082755-78db5a37-4429-414f-8f2c-f88e25bc77cc.png)

1) Implement the basic building blocks of ResNets in a deep neural network using Keras
2) Put together these building blocks to implement and train a state-of-the-art neural network for image classification
3) Implement a skip connection in your network
## 4) Car detection with YOLO
#### FILE NAME: YOLO
YOLO file contains implementation of YOLO algorithem.
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.
![image](https://user-images.githubusercontent.com/86155658/133091576-146950ab-b505-49de-a203-beb0373c9b02.png)
#### References

The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository. The pre-trained weights used in this exercise came from the official YOLO website. 
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/)
