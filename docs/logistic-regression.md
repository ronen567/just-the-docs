---
layout: default
nav_order: 7
title: Logistic Regression
---

# Logistic Regression

## Introduction

This post introduces Logistic regression, an algorithm for performing Binary Classification. The introduction contains 4 chapters:

1. Background: Supervised Learning and Binary Classification
2. Classification Model Selection - why not Linear Regression?
3. Presentetion of Logistic Regression Model
4. Logistic Regression Cost Function
5. Gradient Descent Solution


##  Background: Binary Classification, Supervised Learning


ntroduces the algorithm and is devided to the foll outlines, with an illustration of a simplified classifcation problem. 

assign observations to a discrete set of classes. Some of the examples of classification problems are Email spam or not spam, Online transactions Fraud or not Fraud, Tumor Malignant or Benign. Logistic regression transforms its output using the logistic sigmoid function to return a probability value.

Binary Classification as its name implies, is the operation which assigns observations to one of 2 pre-assigned classes. The classes are conventionally marked by a 1 and 0 indices, where the 1 index will normally be assigned to the positive decision. So for example, decisions such as whether a tumor is malignant or not, or will a customer purchase an item or not, require a Binary Classification. I

Binary Classification belongs to the Supervised Machine Learning category, which model is presented by Figure 1. The Prediction Model resides at the heart of the Learning Model. The predictor's coefficents are calculated during the Training Phase, an later are used for the prediction of data in the normal data phase which follows.
The chaters which follow discusss considerations for the selection of a classifer which fits, presnet the Logistic Regression classifier, the Cost function, and the Grdient Descent algorithm which calculates the predictor's coefficients.


#### Figure 1: Supervise Learning Outlines

![Supervise Learning Outlines](../assets/images/supervised/binary-calssification-machine-learning-model.svg)



## Fitting a Classification Model Selection - why not Linear Regression?

Let's take a Binary Classification simplified example: It is required to predict if a customer will buy a product, based on income. (Note: A good prediction indeed can't base on inome only, and anyway needs much more data points. Still, the simplified example eases the illustration.)

To train the predictor, we use a training data sequence, which consists of labeled data from 11 observations, as presneted by table 1: 

#### Table 1:  Purchase Training Data

|`Income  | Did customer buy? |
|:--------|:------------------|
| 1000    |yes                |
| 1500    |yes                |
| 2000    |yes                |
| 2500    |yes                |
| 3000    |yes                |
| 3500    |yes                |
| 4000    |no                 |
| 4500    |no                 |
| 5000    |no                 |
| 5500    |no                 |
| 6000    |no                 |



Table 1's data is presented on a graph - see Figure 2, where Y=1 means "Customer did purchase". Based on these poinst, it isa needed to perform a predictor which  should be able to make the purchace prediction based on customers' income. We begin by tryin to fit in a Linear Predictor. Will it work?
Figure 3 presents a linear line which was produced by the Linear Regression algorithm. Does it indeed fit? If we set the decision boundery at y=0.5, as illustrated in Figure 4, it seems as if it fits: All the points with income less than 3500 map to 0, and all the rest to 1. But that is an illusion. The Linear Predictor can't realy fit here. Let's show that by taking more observations, as illustrated by Figure 5. Now the line, produced by Linear Regression, maps data points with with an imcome below ~9000 to 0. If we took more points, the classification results might change more, anyway, they can't fir the training data points. Next candicate: Logistic Regression Algorithm!

ense to assign a 1 to the positive hypothesis.

#### Figure 2: Labled Data: customers' income, labeld by Y=1/0 if customer did buy/did not buy

![Supervised  outlines](../assets/images/logistic-regression/binary_point.png)



#### Figure 3: Linear Prediction Mode: Will it fit binary classifcation?

![Linear Prediction for Binary Classification](../assets/images/logistic-regression/binary_point_linear_pred.png)


#### Figure 4: Linear Prediction for Binary Classification with thresholds

![Linear Prediction for Binary Classification THresholds](../assets/images/logistic-regression/binary_point_linear_pred_with_markers.png)




#### Figure 5: Linear Prediction for Binary Classification with thresholds - Problem!

![Linear Prediction for Binary Classification Thresholds Problem](../assets/images/logistic-regression/binary_point_linear_pred_problem.png)



## Logistic Regression Model

The Logistic Regression is a model which predicts the ****probability**** of the hypothesises. The model is based on the sigmoid function, which is presented in Eq. 1 and sketched in Figure 6.

#### Eq. 1: Sigmoid Function

$$\sigma(z)=\frac{1}{1+e^{-z}}$$


#### Figure 6: Sigmoid Function

![Sigmoid Function](../assets/images/logistic-regression/sigmoid-function.png)


#### Sigmoid Properties:

##### $$\sigma(z)_{z \to  -{\infty}} \to 0$$

##### $$\sigma(z)_{z \to  {\infty}} \to 1$$

##### $$\sigma(z)_{z=0}=0.5$$


For Logistic Regression predictor, z argument is replaced by a linear combination of the input dataset x, as shown by Eq. 2:

#### Eq. 2: 
$$z=b+wx$$, 


Plugging  Eq. 2 into Eq 1, results in the Logistic Regression  expression, which is the probability of y=1, given the input vector x and the coefficent set {w,b}.

#### Eq. 3: Logistic Regression Formula

$$p(y=1| x,w,b) = \sigma(b+w^Tx) = \frac{1}{1+e^{^{-(b+w^Tx)}}}$$

Obviously the dependent probability of y=0 is:

#### Eq. 5: Probability of y=0 Regression Formula

$$p(y=0| x,w,b) = 1-p(y=1| x,w,b) = 1- \frac{1}{1+e^{^{-(b+w^Tx)}}}$$



Examining the probailities at the limits and in the middle, we can note that:


$$p(y=1|b+w^Tx \to -{\infty}) \to 0$$


$$p(y=1|b+w^Tx \to {\infty}) \to 1$$


$$p(y=1|b+w^Tx =0 ) = 0.5$$


$$p(y=0|b+w^Tx \to -{\infty}) \to 0$$


$$p(y=0|b+w^Tx \to {\infty}) \to 1$$


$$p(y=0|b+w^Tx =0 ) = 0.5$$


 
 
Next paragraphs we will show how to calculate the Logistic Regression coefficients.



## Finding Logistic Regression Coefficents

In previous posts on Linear Predictor, 2 solutions for for finding the predictor's coefficents were presented:
1. The analytical solution
2. Gradient descent, based on minimizing the Cost function.

Since the predictor's equation is not linear, as it is for the Linear predictor, (reminder: \\(Y=XW+\epsilon\\)), but instead, the predictor's equation is non-linear, where X and W are exponential coefficents, there is no simple analitical solution.
Gradient Descent over a Cost function does work, as we will show soon. Remember that in order to guaranty convergence in a non-local minima, the cost function should be convex. (Remeber what convex is? find explaination here + figure 7 illustrate the idea)

#### Figure 7: CONVEX AND NON CONVEX FUNCTIONS

![Convex and Non Convex Functions](../assets/images/gradient_descent/convex_non_convex.png)

A function is convex if the line between 2 points on the graph are alays above the values of the points between those 2 points


#### Logistic Regression Cost Function

Recalling the cost function used for Linear Refression, the selection of the square error expression as the Cost function might have seemed to be the best and straight forward choice. The square error expression for Logistic Regression is shown in Eq. 6. It is not a convex function, so we are prevented from taking the Square Error expression as our cost function.

#### Eq. 6: square error expression for Logistic Regression


$$SE = \frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}(\hat{y}^i-y^i)^2=\frac{1}{m}\sum_{i=1}^{m}\frac{1}{2}(\frac{1}{1+e^{^{-(b+w^Tx^i)}}}-y^i)^2$$

Instead, the Loss function used is presented in Eq. 7. The detailed development of this expression is presented in the appendix. Note that ***Loss*** function is calculated for a single instance of training data, while Cost is an average of m Loss entries. The superscript i of the loss entries, indicates the index in the traing data sequence. The `log` operator is in natural logarithm. 

Eq. 7a assigns expressions for y=0 and y=1. Eq. 7b combines both equations. Figure 8 illustrate a Loss function, presenting both y=0 and y=1 parts. The behavior of the Loss function is self explained, so I'll not add more on that. The overall Cost function is the sum the m training examples Loss functions, as shown in Eq. 8.  





#### Eq. 7ba: Loss function used for Logistic Regression
$$\begin{cases}
L(b,w)= -log(\hat{y}^i) & \text{ if } y^i=1\\\\\\
L(b,w)= -log(1-\hat{y}^i) & \text{ if } y^i=0
\end{cases}$$




Or expressing it in a single equation:

##### 7b: Loss express in  expressing it in a single equation:

$$L(b,w)=-log(\hat{y}^{(i)})*y^{(i)}$$

$$-log(1-\hat{y}^{(i)})*(1-y^{(i)})$$



Figure 8: Logistic Regression Lost Function


![Convex  Function](../assets/images/logistic-regression/logistic-regression-cost-function.png)



From Eq. 8, we need find the the n+1 coefficients, b and w, whch minimize the cost. Fortunatley, as explained in the Mathematical development section, the Cost function, is concave. This is an important property, otherwise, with local minima poits, it would be harder to find the global minima. But unfortunatley, unlike the Linear Predictor's Cost function, it is not possible to find am analytical solution. Let's have some insight on that:


We need to take the first derivative of the cost function, \\(\frac{\partial }{\partial w_i}J(b,w)\\),  set it to 0 and solve. The derivative formula of Eq. 8 is derived just a few lines ahead, and the result is:

$$\frac{\partial }{\partial w_i}J(b,w)=\frac{1}{m}\sum_{i=1}^{m}(\sigma(b+w^Tx^{(i)}) -y^{(i)})x_i^{(i)}$$

Plugging in:

$$\sigma(b+w^Tx^{(i)}) =  \frac{1}{1+e^{b+wTx^{(i)}}}$$

We get:

$$\frac{\partial }{\partial w_i}J(b,w)=\frac{1}{m}\sum_{i=1}^{m} (\frac{1}{1+e^{b+wTx^{(i)}}} -y^{(i)})x^{(i)}$$

We have a sum of m none linear functions, for which there is no analytical solution, with the exception of special cases with 2 observations, as explained in the paper by Stan Lipovetsky https://www.tandfonline.com/doi/abs/10.1080/02664763.2014.932760.

Instead, we can use a mone alanlytical solution, such as the ****Gradient Descent****. 

Gradient Descent was already explained to the details, and illustrated with the Linear Predictor. So here we can jump directly to implement the solution for Logistic Regression..



Here's the  Gradient Descent operator set on cost function J(b,w), for the free coeffcient {b} and the other linear coefficients {w_j}


#### Eq. 9:  Gradient Descent

#### Eq. 9 a:
$$b:=b-\alpha \frac{\partial J(b,w)}{\partial b}$$

#### Eq. 9b:
$$w_j:=w_j-\alpha \frac{\partial J(b,w)}{\partial w_j}$$
For all {b}, {w_j} j=1...n calculate:


Eq. 9a and 9b for all n coefficents should be repeated iteratively repeated until all {b} and all \\({w_j}\\) converge. The convergence point, is the point where all derivatives are 0, i.e. the minima point. 

The development of the partial derivative \\(\frac{\partial L(b,w)}{\partial w_i}\\), is detailed in the appendix below. The result is presented in Eq 10.


#### Eq 10 a: Cost Function Partial Derivative
$$\frac{\partial }{\partial w_i}J(b,w)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} -y^{(i)})x^{(i)}$$

Pluging $$\hat{y}^{(i)}=\sigma(b+w^Tx^{(i)})$$ into Eq. 10a gives:


#### Eq 10 b: Cost Function Partial Derivative:
$$\frac{\partial }{\partial w_i}J(b,w)=\frac{1}{m}\sum_{i=1}^{m}(\sigma(b+w^Tx^{(i)}) -y^{(i)})x_i^{(i)}$$



Now we are ready to the itterative calculation of \\(w_i, i=1-n\\) and \\(b\\) with Gradient Descent.


Here's the Gradient algorithm procedure:


1. Initialize all n+1 unknow coefficients with an initial value.
2. repeat untill converged: 
   \\ $$b = b - \alpha \frac{1}{m}\sum_{i=1}^{m}(\sigma(b+w^Tx^{(i)}) -y^{(i)})$$
   and for i=1 to n:
   $$w_i = w_i-\alpha \frac{1}{m}\sum_{i=1}^{m}(\sigma(b+w^Tx^{(i)}) -y^{(i)})x_i^{(i)}$$
  

