# Convolutional Neural Network
## 1) Image Segmentation using U-NET
### NOTEBOOK NAME: UNET_Implementation.ipynb
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
### NOTEBOOK NAME: Transfer_learning_with_MobileNet_v2.ipynb
Be using transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier!
![image](https://user-images.githubusercontent.com/86155658/133078539-fe7ffc74-cab7-407b-b85d-deed6b0f64ed.png)
A pre-trained model is a network that's already been trained on a large dataset and saved, which allows you to use it to customize your own model cheaply and efficiently. The one you'll be using, MobileNetV2, was designed to provide fast and computationally efficient performance. It's been pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes.
#### This notebok consists of
1) Create a dataset from a directory
2) Preprocess and augment data using the Sequential API
3) Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
4) Fine-tune a classifier's final layers to improve accuracy 


## 3) Residual Networks
### NOTEBOOK NAME: Residual_Networks.ipynb
Building a very deep convolutional network, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allow you to train much deeper networks than were previously feasible.
Building a ResNet50 model to classify hand signs
![image](https://user-images.githubusercontent.com/86155658/133082755-78db5a37-4429-414f-8f2c-f88e25bc77cc.png)
#### This notebok consists of
1) Implement the basic building blocks of ResNets in a deep neural network using Keras
2) Put together these building blocks to implement and train a state-of-the-art neural network for image classification
3) Implement a skip connection in your network


## 4) Car detection with YOLO
### FILE NAME: YOLO
YOLO file contains implementation of YOLO algorithem.
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.
![image](https://user-images.githubusercontent.com/86155658/133091576-146950ab-b505-49de-a203-beb0373c9b02.png)
#### References
The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's GitHub repository. The pre-trained weights used in this exercise came from the official YOLO website. 
1) Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
2) Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
3) Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
4) The official YOLO website (https://pjreddie.com/darknet/yolo/)



## 5) Face Recognition
### NOTEBOOK NAME: Face_Recognition_.ipynb
Building a face recognition system. Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). In the lecture, you also encountered [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf).

###### Face recognition problems commonly fall into one of two categories: 

**Face Verification** "Is this the claimed person?" For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.

**Face Recognition** "Who is this person?" For example, the video lecture showed a [face recognition video](https://www.youtube.com/watch?v=wr4rx0Spihs) of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem.

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

* Differentiate between face recognition and face verification
* Implement one-shot learning to solve a face recognition problem
* Apply the triplet loss function to learn a network's parameters in the context of face recognition
* Explain how to pose face recognition as a binary classification problem
* Map face images into 128-dimensional encodings using a pretrained model
* Perform face verification and face recognition with these encodings
![image](https://user-images.githubusercontent.com/86155658/133096738-d788fe84-e88e-4897-a535-0ab735d2ff91.png)
#### References
1. Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

2. Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)

3. This implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet

4. Further inspiration was found here: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

5. And here: https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb

## 6) Deep Learning & Art: Neural Style Transfer
### NOTEBOOK NAME: Art_Generation-with_Neural_Style_Transfer.ipynb
Neural Style Transfer, an algorithm created by [Gatys et al. (2015).](https://arxiv.org/abs/1508.06576)
![image](https://user-images.githubusercontent.com/86155658/133117872-f3f24834-28bd-4e23-85f9-950806c8ca9d.png)

1) Implement the neural style transfer algorithm 
2) Generate novel artistic images using your algorithm 
3) Define the style cost function for Neural Style Transfer
4) Define the content cost function for Neural Style Transfer

##### Here are few other examples:

- The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night)
![image](https://user-images.githubusercontent.com/86155658/133117998-6941911e-930d-46fe-8d75-4c440392302a.png)

- The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan.
![image](https://user-images.githubusercontent.com/86155658/133119562-00673a3f-a650-499c-a4e8-fe54b787f42f.png)

- A scientific study of a turbulent fluid with the style of a abstract blue fluid painting.
![image](https://user-images.githubusercontent.com/86155658/133119586-57c0ba8a-13a5-4277-861e-541c81d51ce5.png)


Most of the algorithms optimize a cost function to get a set of parameter values. With Neural Style Transfer, you'll get to optimize a cost function to get pixel values.
#### References
The Neural Style Transfer algorithm was due to Gatys et al. (2015). Harish Narayanan and Github user "log0" also have highly readable write-ups this lab was inspired by. The pre-trained network used in this implementation is a VGG network, which is due to Simonyan and Zisserman (2015). Pre-trained weights were from the work of the MathConvNet team. 

- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) 
- Harish Narayanan, [Convolutional neural networks for artistic style transfer.](https://harishnarayanan.org/writing/artistic-style-transfer/)
- Log0, [TensorFlow Implementation of "A Neural Algorithm of Artistic Style".](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
- Karen Simonyan and Andrew Zisserman (2015). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [MatConvNet.](http://www.vlfeat.org/matconvnet/pretrained/)
