## Internship Project at Center For Cognitive Computing IIIT Allahabad
# Application of Transfer Learning on Hand Written Digits Classification
 Following Research work is submitted to Dr. K.P. Singh, IIIT Allahabad. It may have copyrights on some of it's sections. Referances used for the research are mentioned in footer.

Author: Priyesh Pratap Singh, Shubham Gupta

## 1 Introduction
### 1.1 Problem Statement
The problem requires us to predict the handwritten digit of any person in just one or two samples. So with the help of transfer learning, we aim to build a universal model which can correctly output any handwritten digit written by anyone.
The problem requires us to classify a handwritten digits between 0 to 9 under the limitation of very small available dataset for training. The target dataset consists of only 20 handwritten images of digits.
### 1.2 Motivation
Machine learning and deep learning plays an important role in computer technology and artificial intelligence. With the use of deep learning and machine learning, human effort can be reduced in recognizing, learning, predictions and many more areas. Digit recognition system is the working of a machine to train itself or recognizing the digits from different sources like emails, bank cheque, papers, images, etc. and in different real-world scenarios for online handwriting recognition on computer tablets or system, recognize number plates of vehicles, processing bank cheque amounts, numeric entries in forms filled up by hand (say — tax forms) and so on.
## 2 Background
### 2.1.1 Convolutional Neural Networks
A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution. The final convolution, in turn, often involves backpropagation in order to more accurately weight the end product. Mathematically, it is a sliding dot product or cross- correlation. This has significance for the indices in the matrix as it affects how weight is determined at a specific index point.
![](https://i2.wp.com/sefiks.com/wp-content/uploads/2017/11/cnn-procedure.png?resize=560%2C9999&ssl=1)

### 2.2 SVM
Support Vector Machine (SVM) is machine learning algorithm that analyzes data for classification and regression analysis. SVM is a supervised learning method that looks at data and sorts it into one of two categories. An SVM outputs a map of the sorted data with the margins between the two as far apart as possible. SVMs are used in text categorization, image classification, handwriting recognition and in the sciences. A support vector machine is also known as a support vector network (SVN). More formally, a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.
Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier.

![](https://blog-c7ff.kxcdn.com/blog/wp-content/uploads/2017/02/Margin.png)

### 2.3 Transfer Learning
Transfer learning is a machine learning method in which a model developed for a particular task is reused as a model on another task. Transfer Learning differs from traditional Machine Learning as it is the use of pre-trained models that have been used for another task to kick start the development process on a new task or problem.

![](https://paperswithcode.com/media/tasks/transfer-learning_ZXA3KXi.jpg)

This figure shows how transfer learning is done in neural networks. Here weights of high level ConvNets are freezed and backpropagation is not done in these layers.

### 2.4 Domain Adaptation
Domain adaptation provides an attractive option given that labeled data of similar nature but from a different domain (e.g. synthetic images) are available. As the training progresses, the approach promotes the emergence of “deep” features that are (i) discriminative for the main learning task on the source domain and (ii) invariant with respect to the shift between the domains. This adaptation behaviour can be achieved in almost any feed-forward model by augmenting it with few standard layers and a simple new gradient reversal layer. The resulting augmented architecture can be trained using standard backpropagation.
![](https://miro.medium.com/max/2000/1*uDfooQ7EN9YdSRWM-PWeqw.png)
    This figure shows how domain adaptation is done using a multi output model.

3 Proposed Methodology
Many deep neural networks trained on natural images exhibit a curious phenomenon in common: on the first layer they learn features similar to Gabor filters and color blobs. Such first-layer features appear not to specific to a particular dataset or task but are general in that they are applicable to many datasets and tasks. As finding these standard features on the first layer seems to occur regardless of the exact cost function and natural image dataset, we call these first-layer features general.
In transfer learning we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, that is, suitable to both base and target tasks, instead of being specific to the base task.
As we have to recognize handwritten digits, so we used MNIST as source dataset to extract high level features which will be very similar to high level features of our target dataset.
Since the target dataset is small, it is not a good idea to fine-tune the ConvNet due to the risk of overfitting.
![](https://i.ibb.co/JvKPDBB/figure123.jpg)
A basic representation of our first model


Hence, we, first trained base ConvNet using MNIST dataset then removed the fully connected layers near the end of the pretrained base ConvNet. After training of the base model (pretrained model), we froze the weights and feed it to SVM for classification.Then, We trained our whole model with small target training data using ’rbf’ kernel. In this process only SVM get trained because weights of the ConvNet are frozen.


While experiementing, we built one more model(lets call it second model), by introducing the concept of domain adaptation. Here we forced our ConvNet to learn features of digits’ shapes only, not color or size.For this purpose, we used a multi output model having two set of outputs, one for digit and other for domain classification. Here we have two domains, original MNIST samples and local handwritten digits. Since we want our model to not differentiate the different domains, so we tried to maximize our cost function with respect to domain classifier. And for that, we used gradient reversal technique.
The loss function for our new model is Loss Function = L1λ1 − L2λ2 where L1 is the loss w.r.t to label classifier and L2 is w.r.t to Domain classifier.(λ)1 and (λ)2 are the respective parameters.
Here lambda1 and lambda2 are selected experimentally. For our convince we keep both equal to 1 i.e. both are equidominant.
![](https://i.ibb.co/Bf8TWth/figure124.jpg")
A basic representation of our first model


# 4 Experiments
## 4.1 Experiment Settings
We built our model in keras( github link: https://github.com/prips47/CCC_intern ) with following settings:
1. First Model(By freezing weights of a pre-trained model)
(a) Three Conv2D layers with two MaxPooling layers in between.
(b) Three dense layer were inserted, followed by flattening.
(c) Activation function that we used was relu in hidden layers and softmax function for output layer.
(d) Loss function was categorical cross entropy and optimizer was adam.
2. Second Model(Finetuning with Domain Adaptation)
(a) Three Conv2D layers with two MaxPooling layers in between.
(b) Three dense layer were inserted, followed by flattening.
(c) Activation function that we used was relu in hidden layers and softmax function for output layer.
(d) Loss function(L): L1 (λ)1-L2(λ)2, L1 and L2 are categorical cross entropy loss functions and(λ)1, (λ)2 are parameters. Adam optimizer was used for parameter updation.

## 4.2 Dataset
We have used classical mnist dataset(60,000 training and 10,000 testing samples) for building our pretrained model. We pre processed it by normalising the image vectors and reshaping it so that it can be feed to convolution network. We created our own dataset of 20 samples individually for training and testing on the new model. We recorded our digits and convert it into MNIST format by padding, cropping and normalizing the images.
![](https://i.ibb.co/qWF3jK3/oursample.png)

## 5 Conclusion
In this paper, we present a newly configured model for Handwritten Digit Recognition using Transfer Learning with Domain Adaptation. We achieved accuracy close to 65 %. which can further be improved by using OverSampling Techniques like SMOTE[3] in domain classifier.

## 6 Future Scopes
As mentioned that we didn’t get very good accuracy from our first model and our second model is getting biased because of unbalanced dataset. One can extend these models by performing upsampling and apply various bias-reducing techniques. One can also use well sophisticated pretrained models from autokeras or ludwig.
Humans appear to have mechanisms for deciding when to transfer information, selecting appropriate sources of knowledge, and determining the appropriate level of abstraction. It is not always clear how to make these decisions for a single machine learning algorithm, much less in general.

## 8 References
[1] Lisa Torrey, and Jude Shavlik, “Transfer Learning,” in IEEE/WIC/ACM International Conference on Web Intelligence (WI); 2009.
[2] Yaroslav Ganin, and Vitor Lempitsky, “Unsupervised Domain Adaptation by Backpropogation,” in 15th INTERNATIONAL CONFERENCE ON MALIGNANT LYMPHOMA.
[3] Nitesh V Chawla, Kevin W Boyer, Lawrence O Hall, W Phillip Kegelmeyer, “Synthetic Minority Over Sampling Technique,” in Journal of Artificial Intelligence Research 16(2002) 321-357 .
[4] Jason Yosinski,1 Jeff Clune,2 Yoshua Bengio,3 and Hod Lipson4, “How transferable of Features in Deep Neural Networks,” in NIPS’14 Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2 Pages 3320-3328 .
