#**Traffic Sign Recognition** 


###Data Set Summary & Exploration



I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43



 exploratory visualization of the validation and test set is provided in the html page




As a first step, I decided to convert the images to grayscale because training same images with different colours like the STOP traffic Sign board with Red border and one with green border should be treated the same instead of two different images this can increase your training time

 traffic sign images after  grayscaling is specified in the html page



As a last step, I normalized the image data because  the purpose of normalization is to compare/find relationships between different data.
Normalization in image processing can remove small intensity changes (such as taking photo of the same object but via different angles/lighting).
For example, if we add a small value to each pixel, it becomes a different image, but through normalization, we can essentially treat them as the same data.
This can ensure that the learning model will focus on the structures of the images and not the scale differences between different images.
The unity normalization also has added values of stable gradient feedback for faster learning and better minima.

For example, image 1 has pixel values [1, 2, 3, 4] and image 2 has pixel values [3, 4, 5, 6] (perhaps some glare [2, 2, 2, 2] was added to image 1) and after normalization you will get [0, 1/3, 2/3, 1] for both images. They are treated as same data point, ie, there is no structural difference between the two images. You will still have two images but they are essentially the same when we are trying to find structural differences. Now, hopefully our data set is composed of images with different structure variations and we can use normalization to compare them more effectively.







!

The difference between the original data set and the augmented data set is the following ... 


####2. Model Architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GreyScale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    |
|  RELU	           		|												|
|Max pooling			| 2x2 stride,  outputs 5x5x16 					|
|Flatten				|Output = 400.									|
|RELU    				|												|
|Dropout    			|												|
|RELU    				|												|
|Dropout    			|												|
  




To train the model, I used an AdamOptimizer as i feel this was mentioned to be a better training algorithm than the Stochastic Gradient Descent (SGD). I wish to experiment with other optimizers as well but i couldnt do it due to lack of time .

The batch size which i decided to choose was 200 as my GPU didn't run out of memory and my error optimization didn't take too long 

The number of epochs was chosen such that it is large enough to see the training and validation accuracies saturating ideal which was around 100 but i chose 300 as i did not allow image augmentation .which was the biggest draw back in my model the number of training sample provided is very low i could have augmented and added a 200,000 more replicas of the same data set which would have bought more efficiency in my model .

I applied L2-Regularization after fully connected layer as it prevented overfitting of the data . i wanted to apply drop-out after my convolution layers and maxpooling as well as that would have improved my model but that was consuming too much time to train the model i had used the following reference to apply Regularization
https://arxiv.org/ftp/arxiv/papers/1512/1512.00242.pdf




Number of epochs: 300
Batch size: 170
Optimizer: Adam optimizer 
Learning rate: 5e-3
Dropout keep-prob: 0.5(for training)
Dropout keep-prob: 1.0(for test)


My final model results were:

* validation set accuracy of ?
* test set accuracy of 


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


