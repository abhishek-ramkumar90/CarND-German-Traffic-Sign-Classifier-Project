**Traffic Sign Recognition** 


Data Set Summary & Exploration



I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43



 exploratory visualization of the validation and test set is provided in the Traffic_Sign_Classifier.ipynb	file




As a first step, I decided to convert the images to grayscale because training same images with different colours, like the STOP traffic Sign board with Red border and one with green border, should be treated as  the same instead of two different images ,this can increase your training time thus i have converted this images to gray scale so that it could be treated as the same and our training time could be reduced

 traffic sign images after  grayscaling is specified in Traffic_Sign_Classifier.ipynb	file



As a last step, I normalized the image data because  the purpose of normalization is to compare/find relationships between different data.
Normalization in image processing can remove small intensity changes (such as taking photo of the same object but via different angles/lighting).
For example, if we add a small value to each pixel, it becomes a different image, but through normalization, we can essentially treat them as the same data.
This can ensure that the learning model will focus on the structures of the images and not the scale differences between different images.
The unity normalization also has added values of stable gradient feedback for faster learning and better minima.

For example, image 1 has pixel values [1, 2, 3, 4] and image 2 has pixel values [3, 4, 5, 6] (perhaps some glare [2, 2, 2, 2] was added to image 1) and after normalization you will get [0, 1/3, 2/3, 1] for both images. They are treated as same data point, ie, there is no structural difference between the two images. You will still have two images but they are essentially the same when we are trying to find structural differences. Now, hopefully our data set is composed of images with different structure variations and we can use normalization to compare them more effectively.





2. Model Architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					              | 
|:---------------: |:----------------------------:--------------| 
| Input         		| 32x32x1 GreyScale image   				            	| 
| Convolution 5x5 | 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					       |												                                |
| Max pooling	    | 2x2 stride,  outputs 14x14x6  				         |
| Convolution 5x5	| 1x1 stride, same padding, outputs 10x10x16 |
|  RELU	          |												                                |
|Max pooling			   | 2x2 stride,  outputs 5x5x16 					          |
|Flatten				      |Output = 400.									                      |
|RELU    				     |												                                |
|Dropout    			   |												                                |
|RELU    			     	|												                                |
|Dropout    		   	|												                                |
  




To train the model, I used an AdamOptimizer as i feel this was mentioned to be a better training algorithm than the Stochastic Gradient Descent (SGD). I wish to experiment with other optimizers as well but i couldnt do it due to lack of time .

The batch size which I decided to choose was 200 as my GPU didn't run out of memory and my error optimization didn't take too long 

The number of epochs was chosen such that it is large enough to see the training and validation accuracies saturating ideal which was around 100 but I chose 200 as I did not allow image augmentation ,which was the biggest draw back in my model .The number of training sample provided was very low ,I could have augmented and added a 200,000 more replicas of the same data set which would have bought more efficiency in my model .Due to lack of time i was not able to do it.

I applied L2-Regularization after fully connected layer as it prevented overfitting of the data . i wanted to apply drop-out after my convolution layers and maxpooling as well as that would have improved my model but that was consuming too much time to train the model I had used the following reference to apply Regularization
https://arxiv.org/ftp/arxiv/papers/1512/1512.00242.pdf




Number of epochs: 200
Batch size: 200
Optimizer: Adam optimizer 
Learning rate: 0.001
Dropout keep-prob: 0.5(for training)
Dropout keep-prob: 1.0(for test)


My final model results were:

* validation set accuracy of 0.960
* test set accuracy of 0.941


 

Test a Model on New Images


Their are Six German traffic signs that I found on the web:


Few images might be difficult to classify because I have not done image augmentation as the training samples used are very less and these images might not be available in the training data 


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| kintergarten      		| Keep right   									| 
| road_work    			| Double curve 										|
| Seinforen					| Slippery road											|
| speed_limit_120      		| Speed limit (120km/h)				 				|
| speed_limit_30		| Speed limit (30km/h)     							|
| Stop		| Speed limit (60km/h)     							|


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. 

The accuracy on the captured images is 50% while it was 94% on the testing set thus It seems the model is overfitting


I have also specified the top5 probabilities of the predictions 


