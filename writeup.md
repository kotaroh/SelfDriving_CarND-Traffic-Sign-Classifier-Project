#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./examples/original.png "Original"
[image3]: ./examples/grayscale.png "Grayscaling"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kotaroh/SelfDriving_CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

Here is the summary statistics of the traffic signs data set.

* Training Set:   34799 samples
* Validation Set: 4410 samples
* Test Set:       12630 samples
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes per label, sorted by number of labels in the training data.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the color does not matter for this classification.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image data to reduce the distribution of data from 0 - 255 to 0 - 1.0.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook. Training data is shuffled with shuffle in sklearn.utils later in the traning phase.

To cross validate my model,the separate validation data is used. Also same number of images are used for validation.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 2D     	| 1x1 stride, valid padding, Input = 32x32x1. Output = 28x28x6	|
| RELU					|												|
| Dropout					|		Keep probability 85% for training|
| Max pooling	      	| 2x2 stride, Input = 28x28x6. Output = 14x14x6|
| Convolution 2D     	| 1x1 stride, valid padding, Output = 10x10x16|
| RELU					|												|
| Dropout					|		Keep probability 85% for training|
| Max pooling	      	| 2x2 stride, Input = 10x10x16. Output = 5x5x16|
| Flatten|Input = 5x5x16. Output = 400|
| Fully Connected	| Input = 400. Output = 120|
| RELU					|												|
| Dropout					|		Keep probability 85% for training|
| Fully Connected	| Input = 120. Output = 84|
| RELU					|												|
| Dropout					|		Keep probability 85% for training|
| Fully Connected	| Input = 84. Output = 43|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth cell of the ipython notebook. 

To train the model, I used softmax and cross entropy as cost function, and used Adam optimizer as optimizer for training.

* learning rate 0.01
* One batch includes 256 images
* 20 epochs

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixth cell of the Ipython notebook.

Validation set accuracy as my final model results is  0.946

The first architecture I tried was LeNet implemented in the last lesson. The target image size is the same and the only difference was the number of features as the result of prediction. By using grayscaled data, the depth of input data was also the same. There were multiple problems I encountered.

* The first problem was that the validation accuracy was just about 0.05 or so. After a long investigation, it eas found that value of tf.truncated_normal is too high and  neurons are getting too saturated. By setting up the initial value to 0.1 instead of the default value of 1.0, the accuracy increased dramatically and reached to about 85%.
* At that point, the number of epoch and batch size are set to 10 and 128 respectively. However, the validation process after the training shows that the accuracy did not seem to be saturated. I tried to increase both the number of epoch and batch size and confirmed that the accuracy could be increased to around 0.89.
* To get a higher accuracy, then I introduced dropout layers to the model. Originally keep-probability rate was around 0.5, which did not work very well. I updated it to 0.85 and confirmed we can improve the accuracy by setting this number. It helped the model to increase the accuracy during the whole validation execution process and the model was able to get the accuracy rate 0f 0.946, which is higher than 0.93. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found from the test data set downloaded from the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eighth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited    		| Vehicles over 3.5 metric tons prohibited									| 
| Speed limit (30km/h)    			| Speed limit (30km/h)							|
| Keep right		| Keep right								|
| Turn right ahead      		| Turn right ahead				 				|
|  Right-of-way at the next intersection		|  Right-of-way at the next intersection 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the nineth cell of the Ipython notebook.

For the first image, the model is sure that this is "a Vehicles over 3.5 metric tons prohibited " (feature ID #16). The top five soft max probabilities were;

[9.99995351e-01, 3.45095236e-06,1.04681817e-06, 9.70486838e-08, 2.63701416e-08]

The corresponding id are;

[16,  9, 40,  7, 41]

For the second image, the model is sure that this is "Speed limit (30km/h) " (feature ID #1). The top five soft max probabilities were;

[  9.99641895e-01,   2.93747173e-04,   4.01445614e-05, 1.36849940e-05,   1.02294698e-05]

The corresponding id are;

[1, 2, 0, 5, 4]

For the third image, the model is sure that this is "Keep right	" (feature ID #38). The top five soft max probabilities were;

[  1.00000000e+00,   3.40164116e-12,   5.29262398e-13, 7.74095076e-14,   1.65949223e-15]

The corresponding id are;

[38, 34, 20, 36, 41]

For the forth image, the model is sure that this is " Turn right ahead" (feature ID #33). The top five soft max probabilities were;

[  9.99944091e-01,   3.48959329e-05,   1.53915425e-05,  4.47434468e-06,   4.92537367e-07]

The corresponding id are;

[33, 39, 14, 25, 26]

For the fifth image, the model is sure that this is "Right-of-way at the next intersection" (feature ID #11). The top five soft max probabilities were;
[  9.98846531e-01,   1.13800517e-03,   6.10513507e-06, 2.74995318e-06,   1.66726454e-06]

The corresponding id are;

[11, 30, 27, 40, 21]
