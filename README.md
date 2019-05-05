## Project: Build a Traffic Sign Recognition Program

### Overview
In this project, I used convolutional neural networks to classify traffic signs from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The code about this project is in the file `Traffic_Sign_Classifier.ipynb`

### The Process of this Project
The steps of this project are the following:
* 1. Load the data set
* 2. Explore, summarize and visualize the data set
* 3. Design, train and test a model architecture
* 4. Use the model to make predictions on new images
* 5. Analyze the softmax probabilities of the new images


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

* TensorFlow
  Check the `3 Project Instruction.html` file in `Page` folder to find how to install Tensor Flow.

### Files

`Traffic_Sign_Classifier.ipynb` contains the code for this project.

## Write Up

This write up is to summarize the process and technologies about this project in detail. 

### Step 1. Load the data set

**code:**   `Code Part 1` 


* The pickled data is stored in the folder named `Data`.
  - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
  - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
  - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
  - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.


### Step 2. Explore, summarize and visualize the data set

#### A Basic Summary of the data Set
**Code**: `Code Part 2`

```python
  Number of training examples = 34799
  Number of validation examples = 4410
  Number of testing examples = 12630
  Image data shape = (32, 32, 3)
  Number of classes = 43
```


#### Include an exploratory visualization of the dataset

**Code**: `Code Part 3`, `Code Part 4`

* Plotting traffic sign images, 
* Plotting the count of each sign in the training, validation and test set. 

**Observation:**

As we can see from the images below that the discribution of different classes in training, validation and test set are different.
Some classes have more samples than others. 

![traffic signals](/Image/img1.png)

![traffic signals](/Image/img2.png)

### Step 3: Design and Test a Model Architecture

* Preprocessing images
* Model Architecture
* Model Training
* Solution Approach



#### Preprocessing images

**Code**: `Code Part 5`

* The color-to-grayscale algorithm is Intensity, which calculates the mean of the RGB channels.

* Normalized the grayscale image by `normalized = (gray - 128) / 128`.



#### Model Architecture

**Code**: `Code Part 6`

The Architecture consist of 8 layers:

1. Convolution layer (C1). **Input Size**: 32x32x1; **Filter size:** 5x5; **Output Size**: 28x28x6; **Padding**: Valid; **Stride**: [1, 1, 1, 1]; **Activation Function:** Relu;

2. Max Pooling layer (S2). **Input Size**: 28x28x6; **Output Size**: 14x14x6; **Padding**: Valid; **Pooling Kernel**: [1, 2, 2, 1]; **Stride:** [1,2,2,1];

3. Convolution layer (C3). **Input Size**: 14x14x6; **Filter size:** 5x5; **Output Size**: 10x10x16; **Padding**: Valid; **Stride**: [1, 1, 1, 1]; **Activation Function:** Relu;

4. Max Pooling layer (S4). **Input Size:** 10x10x16; **Output Size:** 5x5x16; **Padding:** Valid; **Pooling Kernel:** [1, 2, 2, 1]; **Stride:** [1,2,2,1]; **DropOut:** 0.5.

5. Flatten layer (F5). **Input Size:** 5x5x16; **Output Size:** 400; **DropOut:** 0.75.

6. Fully connected layer (F6). **Input Size:** 400; **Output Size:** 120; **Activation Function:** Relu; **DropOut:** 0.75.

7. Fully connected layer (F7). **Input Size:** 120; **Output Size:** 84; **Activation Function:** Relu; **DropOut:** 0.75.

8. Fully connected layer (F8). **Input Size:** 84; **Output Size:** 43;

**Why this Architecture:** My work is based on LeNet, and I add three dropout to avoid overfitting. I choose LeNet is because it is a reliable tamplet.



#### Model Training

**Code:** `Code Part 7`, `Code Part 8`, `Code Part 9`, `Code Part 10`

* 1. Set up Model Training Pipeline, `Code Part 7`.
* 2. Set up Model Evaluation Pipeline, `Code Part 8`.
* 3. Train the Model, `Code Part 9`.
* 4. Evaluate the Model, `Code Part 10`.



### Solution Approach

* I tune two parameters (EPOCH and Dropout probability) to find a good model which has validation and test accuracy greater than 0.93. The tuning result as image below shows that **Epoch = 50, Dropout = 0.75** is a good and efficient one.

<img src="/Image/TuneParameters.JPG" width=75% heigth=75%>

* Finally, I  set all the parameter as below: 

Optimizer: AdamOptimizer; 
EPOCHS: 50; 
Batch Size: 128; 
Dropout properbility:0.75 (the same value for all there dropout operation)

* The Validation Set Accuracy at 50th Epochs is 0.958. The Test Set Accuracy is 0.948.

  

### Step 4: Test a Model on New Images

* Acquiring New Images
* Performance on New Images
* Model Certainty - Softmax Probabilities



#### Acquiring New Images

**Code:** `Code Part 11`

Five pictures of German traffic signs from the web is shown below and I use my model to predict the traffic sign type.

<img src="/Image/new_images.png" width=75% heigth=75%>



#### Performance on New Images

**Code:** `Code Part 12`

The accuracy of model prediction for new images is: 0.8 




#### Model Certainty - Softmax Probabilities
**Code:** `Code Part 13`

Calculate the top 5 probability.

<img src="/Image/Top5Probability.png" width=75% heigth=75%>








