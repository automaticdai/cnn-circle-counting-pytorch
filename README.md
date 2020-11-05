# Use CNN to count circles (cnn-circle-counter)

This is a coding challenge based on deep learning with Pytorch. The objective is to count the number of circles in a binary image. Multiple ConV Neural Networks are proposed and implemented as solutions to solve the counting problem. Different network architectures and training datasets are also compared and evaluated. This code is implemented with Python 3.7 and PyTorch 0.4.1.


## Project Organization

- data: contains all the images for training and testing (not included due to the large file size). 
- models: contains saved trained network parameters (only included the best one, again due to large file size). 
- image_gen.py: generate the test images for training, validation and testing.
- models.py: contains all the definitions of CNN models, which includes:
    - AlexNet
    - AlexNet_k7
    - AlexNet_k9
    - Vgg16
    - ResNet18
- main.py: the main function

