
#This is my first 
This is my first post on machine learning and deep learning. After this intro post, more posts are planned which will talk about both theory and implementation issues. Should I start by defining machine learning? Maybe define deep learning too?  So, I’ll take the risk and sum it in a single short sentence,  machine learning, (or I’d better say machine learning algorithms), are computer algorithms which improve automatically through reference from data.

Let me give an example, that might better explain what are machine learning algorithms, and the difference from conventional computer algorithm concepts:
Let’s take the typical image recognition problem: we need to recognize whether a picture is of a cat or not.
A conventional, non machine learning algorithm solution, would use dedicated and complicated computer vision algorithms, which would optionally try to parse the body parts in the image, in the effort to extract cat characterizing features from then. With those collected features, the ‘cat’ or ‘not a cat’ decision would be taken.  As for performance in terms of false detections and miss detections - performance of such algorithms were much poorer than the results of current machine learning algorithms.

On the other hand, a machine learning algorithm model which fit this problem, would be executed in 2 phases:
Training phase - at this stage, the algorithm calculates a set of weight coefficients w, which maximize the likelihood of a stochastic predictor: $$p_{w}(y|x)$$


The prediction phase - at this stage the predicted decision $$y\hat{}$$ is calculated using the predictor calculated during the training phase.

Figure 1 below illustrates the machine learning algorithm operations described above.
Note that in most actual cases, the machine learning algorithm will pass through 3 phases, where a ‘test’ phase is normally added between the training and the read data phases, as illustrated in Figure 2. During the test phase, the error between the predicted result $$y\hat{}$$ and the expected by label y, will be used to decide if the predictor is valid or needs refinements.

More details on the prediction method, how it is calculated, how it works, the 3 phases, etc - in posts which will follow.



Figure 1: Machine learning algorithm 



Figure 2: 3 phases of Machine learning algorithm



So far we tried we illustrated a machine learning algorithm with the cat image recognition task. That task belongs to one category of ML, named Supervised learning. 

Currently the 3 major machine learning categories are:
Supervised learning - currently the most commonly used category
Unsupervised learning
Reinforcement learning.

Let’s briefly describe those 3 categories. We will delve into things in later dedicated posts.


Supervised learning - As can be seen in figures 1 and 2, supervised learning consists of a learning phase with labeled data, just as illustrated with the cat recognition case discussed above.  The reason for naming it ‘supervised’ was the fact that it should be labeled (e.g. as a ‘cat’ or ‘not a cat’) by a knowledgeable supervisor. Data is labeled during phase data too.

The 2 common types of supervised categories are:
Classification
Regression



Classification is the problem of assigning the input data to one of the system’s classes, e.g. the binary classification: ‘a cat’ or ‘not a cat.
Here’s another example of a classification for a supervised machine learning: predict whether a person will buy the newest iphone. - See table 1 below. 


Table 1: Iphone purchase prediction - Binary classification, with structured data


$$x_{1}$$ 
Annual Income ($)
$$x_{2}$$ 
Zip code
$$x_{3}$$ 
House ownership
$$y$$
 Will he buy the newest iphone? 
21200
32321
yes
yes
135000
54322
no
no
243000
43243
yes
yes
320000
63422
yes
no




Each input in table 2 consists of 3 features. The training set consists of 4 labeled examples.



, the input x has 3 features: Annual Income, zip code, and house owning. According to those 3 features {$$x_{1} x_{2} x_{3}$$}, class $$y\hat{}$$ should be predicted, in this case it’s a binary classification - Will the customer buy a new iphone or not. Note that the input data, the ‘features’ can be arranged in a table. This type of data is known as ‘structured’ as opposed to unstructured data, e.g. image input of the previous example.


In Regression supervised learning, the predicted result is not a discrete class, but a continuous value. Table 2 demonstrate a supervised regression machine learning: Predict house prices based on 3 features

Table 3:  House price prediction - regression supervised learning



Number of bedrooms
$$x_{2}$$
Zip code
$$x_{3}$$
floor
$$y$$
Price $
4
32321
2
17900
2
54322
3
21000
2$$x_{1}$$
43243
4
25000
3
63422
1
26000


Note - it is obvious that these 3 features are not enough to predict prices of houses. More features are needed, otherwise, we will see underfitting of results. Overfitting and underfitting will be discussed in future posts.


Next table sums up the 3 supervised learning test cases presented above:





Test Case
Classification Category
Type of data
Prediction type
 Price Prediction
supervised
structured
regression
 Image Recognition
supervised
unstructured
classification
Purchase Prediction
supervised
structured
Binary classification


 Unsupervised Learning

Unlike Supervised Learning, unsupervised learning is used for clustering, and dimensionality reduction  of the data to groups with similar characteristics the machine determines, e.g. similar color of particles in molecular images, similar biochemical structure in viruses or similar probability distribution, which can be used to detect a signal consists noise.  Unlike conventional computer programs, which need to cluster according to apriori known features, now the underlying patterns are discovered by the algorithm.



Figure 3: Clustering









Unlike Supervised learning, as implicit by its name, no labeled training data is needed. This is significant where labeling a large data becomes a complicated task, not to say too complicated when a large scale of training set is required.


How many clusters should the algorithm create? There are some approaches:
Fix number or k clusters
Variable number - find best clustering according to a criteria function.


Let’s see a clustering example, using k-means, one of the most common clustering algorithms, yet one of the simplest. Her k, the number of clusters is assumed apriori. The algorithm searches for the centers of each of the k clusters. Each point of the dataset should be associated to the nearest cluster’s center aka centroid. 

The objective is to minimize the squared error distance function:
$$J=\sum_{j=1}^{k}\sum_{i=1}^{n}\left \|x_{i}-c_{j} \right \|$$

Where J is the cost function, which is the sum of distances between each set point to its related centroid.

So here’s the algorithm:

Randomly select k cluster centers, denoted as $$c_{j}$$. (a popular way to do it -randomly select k data points from the data sets)
Calculate distance between each data point and all cluster centers
Assign each data point to the cluster which distance to its center is minimal
Recalculate cluster centers with:
$$c_{i}=\frac{1}{n}\sum_{j-1}^{n}x_i$$
Recalculate distance between each data point and all cluster centers
Assign the each data point to the cluster which distance to its center is minimal
If no datapoint was reassigned to another cluster, stop. Otherwise, repeat from step 6



---

unsupervised learning categorical data vs continuous

Wiki:
https://en.wikipedia.org/wiki/Unsupervised_learning

 by maximizing some objective function or minimising some loss function. 

whereas supervised learning intends to infer a conditional probability distribution 
{\textstyle p_{X}(x\,|\,y)}
 conditioned on the label 
{\textstyle y}
 of input data; unsupervised learning intends to infer an a priori probability distribution 
{\textstyle p_{X}(x)}




,  images according to shape, colors or any other clustering characteristic the machine determines. 


(From RL book 1.1)

Unlike Supervised Learning, In Unsupervised Learning, aim of prediction is not to classify data to pre-determined hypothesis or forecast a value according to features, but to determine outcomes based of analysis of the data, e.g. clustering images according to shape, colors or any other clustering characteristic the machine determines. 




. 





























































  Reinforcement learning.


Reinforcement learning is used to make decisions sequentially, when input depends on the state of the previous output. Learning is based on 3 concepts: state, action and reward/punishment, as illustrated in the figure below




Finite markovian model



Reinforcement learning is a closed loop problem at which the action influences its next input.The action does not influence the next reward, but also the next state, which again, in a closed loop form, influences on the next action.

The reward is a numerical value, a scalar, 
Given those, the question is: which actions should a software agent take. The agent’s goal should be to get a maximal amount of rewards, without being told which action to take.  Unlike supervised learning which learns from a training set of labeled examples, each identify an hypothesis to which the situation belongs, in RL, the agent reacts to situations which don't belong to a training set, which is  impractical to obtain for all interactive situations which agent has to act. RL is also different from unsupervised learning which typically is about finding structure hidden in collections of unlabeled data. Though RL too does not rely on examples of correct behavior, RL tries to maximaize a reward rather than find hidden patterns or structures in the data)


( the best action to take given the current state. )



Here are examples:

(from book: A mobile robot decides whether it should enter a new room in search of trash to collect, or start his way back to its charging station. Decision is based on current charge level and of how quickly could it find the charger in the padt


 whenin some senses similar to supervised classification, but decisions are made sequentially.


An Introduction to Reinforcement Learning, Sutton and Barto, 1998


From : https://medium.com/analytics-steps/defining-predictive-modeling-in-machine-learning-887c23b7a278


1. Parametric Model
Assumptions are the crucial part of any data model, it not only makes the model easy also improves predictions, so the algorithms that consider assumptions and make the function simple are known as parametric ML algorithms, and a learning model that compiles data with different parameters of a predetermined size, independent to number of training variables, is termed as parametric model.
2. Non-parametric Model
ML algorithms that enable to make strong assumptions in terms of the mapping function are called non-parametric Ml algorithms and without worth assumptions, ML algorithms are available to pick up any functional form training data. Non-parametric models are a good fit for the huge amount of data with no previous knowledge.


