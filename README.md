# Traffic Sign Classification
Build a Traffic Sign Recognition Project
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### First of All
Please refer to this file for the file described in this document  
jpynb:[(./Traffic_Sign_Classifier_190825-latest.ipynb)](./Traffic_Sign_Classifier_190825-latest.ipynb) 
html:[(./Traffic_Sign_Classifier_190825-latest.html)](./Traffic_Sign_Classifier_190825-latest.html) 

# Data Set Summary & Exploration
Used Traffic sign images : [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* The size of training set : 34799
* The size of the validation set : 4410
* The size of test set : 12630
* The shape of a traffic sign image : (32, 32, 3)
* The number of unique classes/labels in the data set : 43  

<img src="./examples/histgram.png"><br/>
<img src="./examples/examples.png"><br/>
43 kinds of signs randomly picked up from the dataset
#### Notes on data samples
There is uneven data in the following
1. Number of data samples (refer the histogram)
2. Size, center position, angle(the sign rotates), view angle(view is not front)
3. Image brightness, saturation and contrast

# Design and Test a Model Architecture
## Preprocessing
#### Counter Measures  
I did the following image processing to take measures against the above three items.
1. Create padding data (add 2 below)
2. Change the original image  
    *Resizing  
    *Center position shift  
    *Rotation  
    *perspective transform  
3. Gray scaling & Normalizing

## Model Architecture
#### LeNet
The model is based on [LeNet](http://yann.lecun.com/exdb/lenet/) by Yann LeCun.
<div style="text-align:center"><br/>
<img src="./examples/lenet.png"><br/> 
Base model by Yann LeCun<br/><br/>

I improved the base models and my final model consisted of the following layers:  

|Layer                       | Description |
|----------------------------|:--------:|
|Input                       | 32x32x1  |
|Convolution (valid, 5x5x24) | 28x28x24 |
|Max Pooling (valid, 2x2)    | 14x14x24 |
|Activation  (ReLU)          | 14x14x24 |
|Convolution (valid, 5x5x64) | 10x10x64 |
|Max Pooling (valid, 2x2)    | 5x5x64   |
|Activation  (ReLU)          | 5x5x64   |
|Flatten                     | 1600     |
|Dense                       | 480      |
|Activation  (ReLU)          | 480      |
|Dense                       | 168      |
|Activation  (ReLU)          | 168      |
|Dense                       | 43       |
|Activation  (Softmax)       | 43       |
## Model Training
Hyperparameters are:

| Name            | Value  | Description                                                 |
|:---------------:|:------:|:-----------------------------------------------------------:|
| `mu`            | 0      | For initilazing Weights with normal distribution            |
| `sigma`         | 0.1    | For initilazing Weights with normal distribution            |
| `learning_rate` | 0.001* | For training neural network (change to 0.0005 on try model) |
| `BATCH_SIZE`    | 256    | Number of images feeding to the model at one time           |

## Solution Approach
1 Change parametor all filters in the convolutional layers and neurons in the dense layers
<img src="./examples/conv_try.png"><br/>


2 Change learning rate
<img src="./examples/rate_try.png"><br/>


3 Change epoch
<img src="./examples/epoch_try.png"><br/>

