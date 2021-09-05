# Deep-Learning
## 1) Image Segmentation using U-NET
#### NOTEBOOK NAME: UNET_Implementation
Building your own U-Net, a type of CNN designed for quick, precise image segmentation, and using it to predict a label for every single pixel in an image - in this case, an image from a self-driving car dataset. 

This type of image classification is called semantic image segmentation. It's similar to object detection in that both ask the question: "What objects are in this image and where in the image are those objects located?," but where object detection labels objects with bounding boxes that may include pixels that aren't part of the object, semantic image segmentation allows you to predict a precise mask for each object in the image by labeling each pixel in the image with its corresponding class. The word “semantic” here refers to what's being shown, so for example the “Car” class is indicated below by the dark blue mask, and "Person" is indicated with a red mask:
![carseg](https://user-images.githubusercontent.com/86155658/132116430-f98b7960-980e-4501-8eb8-4b2970cc55a5.png)\
As you might imagine, region-specific labeling is a pretty crucial consideration for self-driving cars, which require a pixel-perfect understanding of their environment so they can change lanes and avoid other cars, or any number of traffic obstacles that can put peoples' lives in danger.

#### This notebok consists of:
1. Build your own U-Net.\
2. Explain the difference between a regular CNN and a U-net.\
3. Implement semantic image segmentation on the CARLA self-driving car dataset.[https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge].\
4. Apply sparse categorical crossentropy for pixelwise prediction.
