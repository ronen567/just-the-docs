---
layout: default
nav_order: 1
title: Binary Classification and Logistic Regression
---

# Binary Classification and Logistic Regression

This post is about Binary Classification in the context of Supervised Machine Learning Algorithms. In Binary classification, the algorithm needs to select between 2 hypothesis, for a given set of input features. For example: Will a customer will buy the product or not, based on a set of input features which consists of his income, address, occupaition, previous purchaces and so on. So the 2 classes here are: Will purcease and Will not purchace. In the formulas that follow, those 2 classes are denoted as 1 and 0. Which of the 2 classes should take the 1 and which the 0? Bassically, it doesn't matter, but the convention is to assign the 1 to the 'positive' hypothesis and the 0 to the negative hypothesis.

Just note that the focus now is on a Binary Classification, but based on that, we will extend to multi classes. In a post which follows.

So, Question: how will the algorithm work? Answer: It works according to the Supervised Learning algorithm. Let's recap briefly: Figure 1 presents the block diagram of  describe such a system - (re-posting the same diagram for convinience):

Figure 1: Supervise Learning Outlines

![Supervise Learning Outlines](../assets/images/supervised/outlines-of-machine-learning-system-model.svg)


Following Figure 1, we are now concentrating on setting a Prediction Model, which makes the classification decision for the inpuut features X. The model's coefficients are calculated during the Training phase, and later, during the normal data phase, do the classification.

Done with the introduction, the rest of the post will present the following:
1. Present the selection of Logistic Regression as the predictor for Binary Classification.
2. Present the usage of Gradient Decent to find the Logistic Regression coefficients


To start with, we will use a reduced features order Binary Classification sceanrio, i.e.: Predict if a customer will buy a new product based on his income. Such a prediction maybe does not make sense practicdally, but the reduced features dimensions to 1, makes it easier to present it graphically.
So now, Figure 2 presents the purchase perdiction based on income only.



Figure 2:  Binary Classification - Purchace prediction based on income


![Supervised  outlines](../assets/images/logistic-regression/binary-classification-points.png)


The data set presented in Figure 2, hold the labeled training data, i.e. the each income feature point is labeled with its related purchase decisoin taken. We would need a prediction model that can fir these points. According to our paradigm, after fitting the model to the training data, it should fir the normal unlabeled data that comes after. (Though after training, the model's performance should be tested before accepted to be used for real data).

So which model can we take? We previously reviewed the linear prediction model, used for predicting Regression data. Will it fit here too?  Let's see. Figure 3 

examines Linear Prediction for Binary Classification.


![Linear Prediction for Binary Classification](../assets/images/logistic-regression/linear-prediction-binary-classification.png)


Figure 3 presents a line which is assumed to model the data points. Examine the model: According to it, a income of 3000, which was labeld with a 0, will now be mapped to ~0.4. and the income of 3500 now maps to 0.5. Can this model create valid predictions? Figure 4 presnets the decision boundary - any point on the line, from 0.5 and up will be classified as 1, and the rest as 0. One might tend to think that this model is valid - but that is not correct. Justtake another set of training points, as presented by Figure 5, keep the thershold on 0.5, and now, only income from 5000 and up is mapped to 1. And if we had taken more points, obviously the treshhold would move more. 

Figure 4: Linear Prediction for Binary Classification with thresholds

![Linear Prediction for Binary Classification THresholds](../assets/images/logistic-regression/linear-prediction-binary-classification_thresholds.png)


Figure 5: Linear Prediction for Binary Classification with thresholds - Problem!

![Linear Prediction for Binary Classification Thresholds Problem](../assets/images/logistic-regression/linear-prediction-binary-classification_thresholds_problem.png)



Obviously, linear prediction doesn't work for Binary Classification. Another different prediction model is needed. Let me present the Logistic Regression Model.


## Logistic Regression Model

The Logistic Regression is currently the most common predictor model used for binary classification. The model is based on the sigmoid function presented in Eq. 1.

Eq. 1: Sigmoid Function

$$g(z)=\frac{1}{1+e^{^{-z}}}$$


The sigmoid function's values are in the range [0,1], as shown in Figure 5

Figure 5: Sigmoid Function

![Sigmoid Function](../assets/images/logistic-regression/sigmoid-function.png)


The Logistic Regression predictor is based on Eq. 1, but with pluging $$z=b+wx$$ in, where $$x=\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4 \\..\\x_n 
\end{bmatrix}$$, an n dimensional vector, is an input data set, AKA input features, and $${b,w}$$ are the predictor's coefficients, such that b is a scalar and w=\begin{bmatrix}
w_1 \\ w_2 \\ w_3 \\ w_4 \\..\\w_n 
\end{bmatrix} is the vector of coeffcients. Pluging $$b+wx$$ into Eq. 1 gives:

Plugin $$b+$wx$$ into Eq 1 gives Eq. 2.

Eq. 2: Logistic Regression Formula

$$h(b+w^Tx)=\frac{1}{1+e^{^{-(b+w^Tx)}}}$$


Now, if we had the values of b and w (we will show how to calculate those soon), and the input features set x, we could calulate the predicted value $$h(b+w^Tx)$$

As shown by Figure 5, $$h(b+w^Tx)$$ is bounded in the range [0,1], so our predicted value will be in that range. So, we will not predict just the neares hypothesis, but the probabilty of the hypothesis y=1, i.e. we will calculate $$p(y=1, x)$$, which means: the probability that y=1, given the input x. It can be easily seen that  $$p(y=1, x)$$ tends to 1, as $$x \to \infty $$, and $$p(y=1, x)$$ tends to 0, as $$x \to 0 $$. The point z=0 om Figure 5, would be classified to x=1 with probability of 50%. z=2.5 would be classified to 0.1ת i.e. probability of 10% to be 1, which is equivalent to probability of 90% to be 0.

Having selected the prediction model, to calculate the predictions with Eq. 2, we need to have the coefficents. What is the criteria for the selection of the coefficients' values? Answer is trivial - we will take the coefficients that make the model predict values which are the closest to the actual values. (This is done in the Traingning phase, of course, with labeled data).

To find the closest prediction, we will define a cost function which expresses the error between the labeled points actual values, and their values predicted by the model. We will select the coefficients which minimize this cost. So here we go.


#### Logistic Regression Cost Function

Eq. 2 determines the prediction function. It's a function of the input features x, and coefficients b and w.
So, in order to be able to estimate the probability of y=1 for a given input x, as expressed by Eq. 2, we need to calculate the coefficients {b, w} as expressed in Eq. 3. It shows the n dimension vector prodcut expression that should be plugged into Eq. 2.


Eq. 3: Logistic Regrerssion n dimensional coefficients

$$ b+wx = b + w_1x_1+w_2x_2+....w_nx_n $$

So, how should the values of parameters {b, w} be determined? Or in other words, which {b, w} make the best predictor? To answer this, we need to define the predictor's required charectaristics. Let's do it: We need to define a cost function, which expresses the diffeerence between the algorithm's predicted values $$\hat{y})$$ and the actual $${y}$$ values. Having such a cost function, we will find the set of coefficients which minimizes that cost function.

Reminder: For Supervised Regression, we presented the Linear Regression predictor. The predictor is trained with labeld data during the Training phase, using a set of m labeled data points. The cost function used there, was the Euclidean distance between the data points' real value (the labels), and the model's predicted points, as expressed by Rq. 4.

Eq 4: Cost function - Euclidean Distance:

J(b,w) = \frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}(h_{b,w}(x^i)-y^i)^2


Having Eq. 4, we showed 2 optional solutio: The analytic solutiuon and the Gradient Descent solution. question now is - can the cost function expressed in Eq.4 be used for Logistic Regression predictor as well? Answer is no. Reason: The analtical solution is too complex, and the Gradient descent may not converge, as this cost function (Eq. 4), when Gradient Descent is plugged in, is not convex - see Figure 6, i.e. it has many local minimas. When using Gradient Descent to find the minima of such a function, it may find a local minima but not the global one. So, we need a convex cost function. 

Figure 6: Non Convex Cost Function




![Non Convex Cost Function](../assets/images/logistic-regression/non-convex-function.png)


Figure 6: A Convex  Function


![Convex  Function](../assets/images/logistic-regression/convex-function.png)



Eq. 5 presents the cost function used for Logistic Regression


#### Eq. 5: Cost function used for Logistic Regression
5.a

$$Cost(h_{b,w}(x^i,y^i))=\left\{\begin{matrix}
-log (h_{b,w}(x^i,y^i)) \; if\; y=1\\
-log (1-h_{b,w}(x^i,y^i))\; if \;y=0
\end{matrix}\right.
$$

Or expressing it in a single equation:

4.b
$$Cost(h_{b,w}(x^i), y^i)=[y^ilog(h_{(b,w)}(x^i))+(1-y^i)log(1-h_{(b,w)}(x^i))]$$

The index $$i$$ relates to the $$i^{th}$$ example out of m training examples.
Let's examine the cost function optinal outputs, at the 4 'extreme' points:

If hypothesis is y=1 and also predicted probability p(y=1|x) is 1 (i.e. the probability of y=1 given x is 1), then the cost is 0. ($$Note that log(1)=0$$).
If hypothesis is y=0 and also predicted probability p(y=1|x) is 0,ithen the cost is 0. ($$log(0)=1$$).


If candidate hypothesis is y=1 but prediction probability p(y=1|x) is 0, then the cost is $$\infty $$. ($$-log(0)=\infty $$).
If candidate hypothesis is y=0 and prediction probability p(y=1|x) is 1, then the cost is $$\infty $$. ($$log(0)=1$$).

So far, the cost function behavior does make sense. Let's plot the cost function, and gain some visibility on it.



Figure 7: Logistic Regression Cost Function


![Convex  Function](../assets/images/logistic-regression/logistic-regression-cost-function.png)


The overall cost function is the sum the m training examples cost functions:

#### Eq. 6: Logistic Regression overall Cost Function

$$
J(b,w)=\frac{1}{m}\sum_{i=1}^{m}Cost(h_{b,w}(x^i), y^i)=\\
-\frac{1}{m}\sum_{i=1}^{m}[y^ilog(h_{(b,w)}(x^i))+(1-y^i)log(1-h_{(b,w)}(x^i))]
$$

Where

$$h_{(b,w)}=\frac{1}{1+e^{-(b+w^Tx)}}$$


With the cost function at hand, we need to find the set of coefficients {b,w) which minimizes the cost.

In the linear prediction example, we considered 2 solutions:
1. The "Analytic Solution" at which the cost we took partial derivatives of the Lossfunction per each of the n+1 coefficients. Each derivative was set to 0, so we had n+1 normal equations which could be analytically soved.

2. Gradient Descent


Here solving the normal equations is by far more complex, still, we can use Gradient Descent.


So here's the Gradient Descent formula:

#### Eq. 7: Gradient Descent For J(w,b)
Repeat till convergence:
#### Eq. 7a:

$$b:=b-\alpha \frac{\partial J(b,w)}{\partial b}$$

#### Eq. 7b:
For all {b}, {w_j} j=1...n calculate:

$$w_j:=w_j-\alpha \frac{\partial J(b,w)}{\partial w_j}$$

Explaination for the Gradient Decent Process :
n the Gradient Descent solution, Eq. 7a and 7b should be repeatedly computed, where at each iteration a new set of {b,w} parameters are computed. The iterative process should contiue until b and w converge, i.e. the partial derivatives are 0. As we know, the derivatives are 0 at the minima. So the computed (b,w) are the parameters that minimize the cost function, as required.


Let's calculate the partial derivative $$\frac{\partial J(b,w)}{\partial b}$$:

RONEN TILL HERE!!!!!!!!!!

To simplify the derivation, we'll base on the derivatives chain rule. Before getting to that, let's prepare arrange the cost function:


We denote:
x=b+wx
and h

Eq. 8:

$$\frac{\partial f(y)}{\partial x}=\frac{\partial f(y)}{\partial y}\frac{\partial y}{\partial x}$$





$$


og the .
Cost(

Is is given in Eq. 5.


to find a minima with Gradient Descent, 



Having the prediction function, we need to calculate its coefficients. 

Similarily to Linear Regression, we will determine a cost function, whi



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

