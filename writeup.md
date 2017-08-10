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

[testimg1]: ./examples/no_passing.jpg "No passing"
[testimg2]: ./examples/pedestrians.jpg "Pedestrians"
[testimg3]: ./examples/priority_road.jpg "Priority Road"
[testimg4]: ./examples/seventy.jpg "70 km/h"
[testimg5]: ./examples/beware_of_ice_snow.jpg "Beware of ice/snow"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below is a histogram showing how the number of training samples per class. It is clear from the visualization that the number per class is uneven, which may make the convolutional neural network more easily recognize some classes more than the others.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The data was preprocessed with standard techniques. First the mean image across the training set only is computed. Then the standard deviation of all the pixel intensity values in the training set is calculated. The computed mean image is subtracted from the training, validation, and test images, and then each image's pixel intensity values are divdied by the computed standard deviation.

The above technique serves to center and normalize the training data so that convergence, and thus training, is faster. The pixel values were in the range of [0, 255], and after preprocessing, are in the range of real numbers. Because the values in the dataset are no longer just positive values, during backpropagation, the gradient on the weights doesn't simply consist of only positive or only negative numbers. This means that the weights aren't updated in only the positive or negative direction. In two dimensions, if you visualize the weights being updated only in the positive direction and then in the negative direction, the path to the optimal point that results in the lowest loss value is a zigzag. A straighter line would be more efficient, which is why centering the data and preventing only positive or only negative gradients during each iteration of backpropagation speeds up convergence. The data is normalized to reduce the likelihood of big weight updates. This means a higher learning rate can be used.




![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Dropout    	      	|                                 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32        			|
| Flatten               | outputs 400                                   | 
| Fully connected 		| outputs 120        							|
| RELU					|												|
| Dropout    	      	|                                 				|
| Fully connected 		| outputs 84        							|
| RELU					|												|
| Dropout    	      	|                                 				|
| Fully connected 		| outputs 10        							|
| Softmax				|            									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer, a batch size of 128, and 120 epochs. The learning rates are 0.001, 0.0005, and 0.0004. The learning rate starts out as 0.001, and as soon as the difference between the current and previous validation accuracies is negative, the next learning rate is used. The final learning rate of 0.0004 is used until the end. The weights are initialized with values within two standard deviations of the mean in a normal distribution, with the standard deviation = 0.1 and the mean = 0. The biases are initialized with the value of zero.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started out with the architecture used in the LeNet lab. My hypothesis was this architecture should be sufficient to obtain a validation set accuracy of at least 0.93 because the traffic sign images are 32x32x3, not that much bigger in terms of width and height from the MNIST dataset. I started out using a learning rate of 0.001, batch size of 128, and 20 epochs to train the classifier. This resulted in 0.891 validation set accuracy. Because there are 43 classes instead of 10 like in MNIST, I decided to increase the capacity of the network by doubling the number of filters in the first convolution layer. I chose to double because my goal was to use an unbounded binary search to find a good number. Fortunately, this doubling bumped up the validation set accuracy. I then doubled the number of filters in the second convolution layer. This also helped. After this, I noticed the training set accuracy approached 1.0 after several epochs whereas the validation set accuracy didn't climb nearly as high and even dropped at times. This was a sign that my classifier was overfitting so I borrowed the idea of using dropout after the fully connected layers from the AlexNet paper. As expected, this closed the gap between the training and validation set accuracies because dropout essentially trains an ensemble of classifiers and averages out the predictions of the ensemble. Dropout also reduces a neuron's dependence on the presence of another neuron, which increases regularization. I decided to add dropout after the convolution layers as well, which further decreased the gap. As expected, this slowed down convergence so I bumped up the number of epochs multiple times. I knew I could keep increasing the number of epochs by looking at the graph of the validation set accuracy versus the number of epochs. If the validation set accuracy showed a slight trend upward, I knew there was more training left to do. Because the number of epochs had increased so much, I wanted to squeeze as much of the network as possible so I experimented with various learning rates until I arrived at 0.0004. In the beginning of training, such a low learning rate is not needed, so I started with 0.001, which decreases to 0.0005, and finally ends up at 0.0004 using the method I described earlier.

Using this method, I achieved a 0.952 validation set accuracy, and I felt comfortable enough about the network not having overfitted too much.
My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.952
* test set accuracy of 0.938

Given these results, I'm sure much more can be achieved with this modified LeNet architecture. For example, I didn't experiment with transforming the images into grayscale. Furthermore, I didn't augment the data.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][testimg1] ![alt text][testimg2] ![alt text][testimg3] 
![alt text][testimg4] ![alt text][testimg5]

The first image might be difficult to classify because the photo wasn't taken from straight ahead of it. Since I didn't augment the data with any skewing, my classifier might not be able to classify it properly. The same idea applies to the other images. Most of them are pretty bright too, while the training set images are relatively low in brightness. Furthermore, these images are cropped tightly around the sign, whereas the training set images aren't always cropped tightly around the signs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Speed limit (70km/h)   		    			| 
| Pedestrians        	| Right-of-way at the next intersection 		|
| Beware of ice/snow	| Children crossing								|
| Priority road	    	| Priority road					 				|
| No passing			| No passing        							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This does not compare favorably to the accuracy on the test set of 93.8%. However, this is due to this set of 5 signs being rather small compared to the 12,630 images in the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For all images, the classifier is almost 100% certain that it has chosen the right answer. The following are the top 5 predictions. The asterisk next to the sign type denotes the correct label. The only image the classifier got wrong was "Beware of ice/snow", which it predicted most likely as "Bicycles crossing".


Probability                   Prediction          
1.0                           * Speed limit (70km/h)
1.2198311208789364e-13        Speed limit (30km/h)
2.956811706415923e-20         Speed limit (20km/h)
0.0                           Speed limit (50km/h)
0.0                           Speed limit (60km/h)

For the second image ... 
Probability                   Prediction
1.0                           * Pedestrians         
0.0                           Speed limit (20km/h)
0.0                           Speed limit (30km/h)
0.0                           Speed limit (50km/h)
0.0                           Speed limit (60km/h)

Probability                   Prediction
0.9999997615814209            Bicycles crossing   
2.199468127628279e-07         Road work           
9.620956445415018e-20         * Beware of ice/snow     
5.05926311913774e-27          Children crossing   
0.0                           Speed limit (20km/h)

Probability                   Prediction
1.0                           * Priority road       
0.0                           Speed limit (20km/h)
0.0                           Speed limit (30km/h)
0.0                           Speed limit (50km/h)
0.0                           Speed limit (60km/h)

Probability                   Prediction
1.0                           * No passing          
0.0                           Speed limit (20km/h)
0.0                           Speed limit (30km/h)
0.0                           Speed limit (50km/h)
0.0                           Speed limit (60km/h)
