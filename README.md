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

## Data Set Summary & Exploration
Used Traffic sign images : [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* The size of training set : 34799
* The size of the validation set : 4410
* The size of test set : 12630
* The shape of a traffic sign image : (32, 32, 3)
* The number of unique classes/labels in the data set : 43  

<img src="./examples/histgram.png"><br/>
<img src="./examples/examples.png"><br/>
.                       43 images randomly picked up from the dataset
#### Notes on data samples
There is uneven data in the following
1. Number of data samples (refer the histogram)
2. Size, center position, angle, view angle
3. Image brightness, saturation and contrast

#### Counter Measures  
I took the following image processing measures.
1. Create padding data (add 2 below)
2. Change the original image  
    *Resizing  
    *Center position shift  
    *Rotation  
    *perspective transform  
3. Gray scaling & Normalizing

# Design and Test a Model Architecture
## Preprocessing


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

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

