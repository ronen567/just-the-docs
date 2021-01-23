# Binary Classification and Logistic Regression

This post is about Supervised Classification problems, at which it is needed to predict a descret value, e.g. predict if a customer will buy the product or not, or predict whether a tumor is benign or malignant. 
We denote the value y of the binary classes as 0 and 1, so we assine the value y=0 to denote the case at which the customer does not purchase, or the tumor is benign, and accordingly y=1 for the case customer does purchase or tumor is malignant.


So let's follow our example of predicting customers' purchases based on their income. (This is a simplified for the sake of the example, even if technically the income feature is clearly not sufficient to predict a purchase).

Figure 2:  Binary Classification - Purchace as a function of income


![Supervised  outlines](../assets/images/logistic-regression/binary-classification-points.png)


Can linear Prediction model this data? Look at Figure 3.

Figure 3: Linear Prediction for Binary Classification


![Linear Prediction for Binary Classification](../assets/images/logistic-regression/linear-prediction-binary-classification.png)


Looking at Figure 3, one might tend to think that linear prediction might work for binary classification also. Look at figure 4: the threshold is at 0.5, and anything to its right is indeed 1. 

Figure 4: Linear Prediction for Binary Classification with thresholds

![Linear Prediction for Binary Classification THresholds](../assets/images/logistic-regression/linear-prediction-binary-classification_thresholds.png)


But the above graphical presentaions are missleading. Just have more points, and the linear prediction now changes, as presented by Figure 5.

Figure 5: Linear Prediction for Binary Classification with thresholds - Problem!

![Linear Prediction for Binary Classification Thresholds Problem](../assets/images/logistic-regression/linear-prediction-binary-classification_thresholds_problem.png)




Figur 5 obviously shows that a linear predictor can't be used for binary classifcation. So another different prediction model is needed.






Logostic Regression is currently one of the most commom prediction model algorithm used by Machine Learning algorithms for binary classification. In case you're not familiar with prediction models, and how to solve for their coefficients, or even in case you have no clue about what prediction am I talking, I suggest you read my post on that before. Not mandatory though. If the term "Binary Classification" needs clarifications, I'd start with my Intro to Machine Learning. Not mandatory though.

In any case, to start with, I posted here again the Supervised Machine Learning blog diagram.

#### Figure 1: Supervised Machine Learning blog diagram

![Supervised learning outlines](../assets/images/supervised/outlines-of-machine-learning-system-model.svg)

As Figure 1 shows, the predictor sits in the heart of the system. The perdictor's coeficients are calculated during the Training phase, then ready to use in the Testing and Normal Data phases.

This post explains the Logistic Regression model, and the developemnt of the model's parameters' solution. We'll walk top to bottom - begin with an illustration of a classification problem, then present the Logistic Regression predictor model, and eventually show how to find its coefficients, with the good-old Descent Regression algorithm.


Here's the (commonly used) binary classification example: It is needed to predict whether a tumor is benign or maligent, based on its size. 




multi class



we need to predict if a customer will buy Iphone as his next phone, based on the next smartphone of a customer , which already owns a smatphone,  will buy another iphone after 2 years, based on whether he owned an iphone before or not. We'll see that normally a 
. This example is not realy realistic, but we'll use it to start with binary prediction based on a single feature. So, suppose we need to predict if a customer, which already owns a smatphone,  will buy an iphoe, based on whether he owned an iphone before or not. Suppose  that if he owned one 

