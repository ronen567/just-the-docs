---
layout: default
nav_order: 6
title: Backwards Propogation
---
# Backwards Propogation

## Introduction

This is the third in series of 3 deep learning intro posts:
1. Introduction to Deep Learning which introduces the Deep Learning technology background, and presents network's building blocks and terms.
2. Forward Propogation, which presents the mathematical equations of the prediction path.
3. Backward Propogation which presents the mathematical equations for network's coefficents calculation, done during the training phase.

In prediction mode, as described in the previous post, the network and its coefficients are static, while only the input data changes. Given that, the predicted value is caculated using the forwarding equations presented. The Training AKA fitting mode is different: A set of example data is sent constantly, while the network's coefficients are modified till their values converge to an optimal value.
In this post we will develop the equations for calculating the neural network's optimized weights and biases using backwards propogation. 

## Feed Forward Recap

To recap the Feed Forward equation, Figure 1 taken from the previous post is posted here again. It presents the operators and parameters which run the data input journey through the network, till the predicted value \\(hat{y}\\) is generated. Each layer consists of an activaton operator \\(g(z)^{[l]}\\), and a matrix of weighting coefficients \\(\bar{w}^{[l]}\\), and a bias vector  \\(\bar{b}^{[l]}\\). 

### Figure 1: Feed Forward Flow
![neuron_cascaded_operator](../assets/images/neural-networks/forwarding-vectorized-flow.png)


As Figure 1 shows, each layer consists of 2 cascaded operators, the first calculates the sum of the weighted input and the bias, and the second is a non linear activation function. Eq. 1 presents the above 2 functions expressed for any layer l.

### Eq. 1: Feed Forwarding Equations Layer l
#### Eq. 1a: Feed Forwarding Equations Layer l - Weighted input
 $$
 \bar{z}^{[l]}=\bar{w}^{[l]}\bar{a}^{[l-1]}+\bar{b}^{[l]}
 $$
 
#### Eq. 1b: Feed Forwarding Equations Layer l - activation

$$a^{[l]}=
g^{[l]}(z^{[l]})$$

To conclude this recap from previous post, Eq.2 expresses \\(a^{[l]}=\\) Eq. 1b in a combined form which composes all previous layers:

### Eq. 2: Feed Forward Combined Equation

\\(A^{[l]}=g^{[l]}(w^{[l]}g^{[l-1]}(w^{[l-1]}g^{[l-2]}(w^{[l-2]}g^{[l-3]}(......g^{[1]}(w^{[1]}A^{[0]}))))\\)




## Cost Function and Gradeint Descent

Here we start the journey of finding the optimized network's parameters. 


Given the input-output set (x,y) and the prediction result \\(\hat{y}\\), a cost function measures the difference between the true values and the model's prediction results. The set optimized set of parameters we search for, is the one the minimizes a cost function.

We already met 3 types of Cost functions as shown in Eq. 3:

### Eq. 3: Cost Functions
#### Eq. 3a: MSE (Mean Squared Error) Cost Function

\\(C(w,b)=\frac{1}{2m}\sum_{j=1}^{m}\left \| y-\hat{y} \right \|^2\\)

#### Eq. 3b: MAE (Mean Absolute Error) Cost Function

\\(C(w,b)=\frac{1}{2m}\sum_{j=1}^{m}\left | y-\hat{y} \right |^2\\)

#### Eq. 3c: Cross Entropy Cost Function - Used for Logistic Regression

\\(C(b,w) =-\sum_{i=1}^{m}[y_i^{(i)}log(\hat{y}^{(i)})+(1-y^{(i)})log(1-\hat{y}^{(i)})]\\)

**Just a side note to prevent confusion between Loss and Cost functions** - **Loss** Function measures the difference between true value (y), and the model's prediction results (\\(\hat{y}\\)), while **Cost** function is the average over a batch of m Losses, e.g.  avarage over the entire training sequence of over a partial batch of it.


With a selected cost function at hand, we need to find the optimizing parameters set. We will use the Gradient Descent algorithm. Note that Gradient Descent has some commonly used variants such as tochastic gradient descent and ADAM. we will cover those algorithms in an exclusive post.

If you're not familiar with Gradient Descent, it is suggested you read the posts on Gradient descent before.

The equations for determining the coefficients are presented by Eq. 4, Where the superscript [l], l=[1,L], denotes the layer, and \\(\alpha\\) is the learning rate.

### Eq 4: Gradient Descent Recursive Equations
#### Eq 4a: Weight Recursive Equation
$$
w^{[l]}=w^{[l]}-\alpha\frac{\partial C}{\partial w^{[l]}}
$$
#### Eq 4b: Bias Recursive Equation

$$
b^{[l]}=b^{[l]}-\alpha\frac{\partial C}{\partial b^{[l]}}
$$


As Eq. 4 shows, we need to find the Cost function's derivatives, with respect to all layers' coeffcients. 

To do that, we will use the Backward propogation algorithm, as explained next.


## Backwards Propogation Algorithm

Our challenge is to find the derivatives of the Cost function with respect to all weight and bias, i.e. find \\(\frac{\partial C}{\partial w^{[l]}}
\\) and \\(\frac{\partial C}{\partial b^{[l]}}\\), for 1<=l<=L.

Let's start with finding the partial derivative of C with respect ot last layer's, (i.e. the output layer),  coefficents, i.e. find \\(\frac{\partial C}{\partial w^{[L]}}
\\) and \\(\frac{\partial C}{\partial b^{[L]}}\\). (L is of course last layer's index). 
For convinience, let's re-write Eq.1, layer's equations, for layer L:


 ### Eq. 5a: Layer L Weighted input
 $$
 \bar{z}^{[L]}=\bar{w}^{[L]}\bar{a}^{[L-1]}+\bar{b}^{[L]}
 $$
 
 ### Eq. 5b: Layer L activation

$$a^{[L]}=
g^{[L]}(z^{[L]})$$



Since \\(C(y,\hat{y})\\) is obviously a function of \\(\hat{y}\\), and \\(\hat{y}=a^{[L]}\\), we can use the derivative chain rule and write the following derivative equations:

 ### Eq. 6: Cost Derivatives with respect to layer L parameters

 ### Eq. 6a: Cost Derivatives with respect to weights
\\(\frac{\partial C}{\partial w^{[L]}}=\frac{\partial C}{\partial a^{[L]}}* \frac{\partial a^{[L]}}{\partial z^{[L]}}* \frac{\partial z^{[L]}}{\partial w^{[L]}}\\)

 ### Eq. 6b: Cost Derivatives with respect to bias

\\(\frac{\partial C}{\partial b^{[L]}}=\frac{\partial C}{\partial a^{[L]}}* \frac{\partial a^{[L]}}{\partial z^{[L]}}* \frac{\partial z^{[L]}}{\partial b^{[L]}}\\)


Both Eq. 6a and Eq. 6b consist of a chain of 3 partial derivatives:
1. \\(\frac{\partial C}{\partial a^{[L]}}\\) - This derivative depends on selected Cost function. Find 3 of the most common cost functions are listed in Eq. 3. Find detailed derivatives equation for commonly used Cost functions in appendix.
2. \\(\frac{\partial a^{[L]}}{\partial z^{[L]}}\\) - This derivative depends on activation function. Find detailed derivatives equation for commonly used activation function in appendix.
3. \\(\frac{\partial z^{[L]}}{\partial w^{[L]}}\\) - Pluging in Eq. 5a gives: \\(\frac{\partial z^{[L]}}{\partial w^{[L]}}\\)=\\(\frac{\bar{w}^{[L]}\bar{a}^{[L-1]}+\bar{b}^{[L]}}{\partial w^{[L]}}= \bar{a}^{[L-1]}\\)
4. \\(\frac{\partial z^{[L]}}{\partial b^{[L]}}\\) - Pluging in Eq. 5b gives: \\(\frac{\partial z^{[L]}}{\partial b^{[L]}}\\)=\\(\frac{\bar{w}^{[L]}\bar{a}^{[L-1]}+\bar{b}^{[L]}}{\partial \bar{b^{[L]}}}=1\\)

Following that, the expressions for Cost derivatives for layer L are now:


### Eq. 7: Cost Function Derivative for Layer L
#### Eq. 7a: Cost Function Derivative with respect to weights

\\(\frac{\partial C}{\partial w^{[L]}}=\frac{\partial C}{\partial a^{[L]}} * g^{'[L]} * \bar{a}^{[L-1]}\\)

#### Eq. 7b: Cost Function Derivative with respect to bias
\\(\frac{\partial C}{\partial b^{[L]}}=\frac{\partial C}{\partial a^{[L]}}*g^{'[L]}\\)



Let's continue the stepping backwards to Layer L-1, keeping up with the derivative chain rules:

\\(\frac{\partial C}{\partial a^{[L-1]}}=\frac{\partial C}{\partial z^{[L]}} * \frac{\partial z^{[L]}}{\partial a^{[L-1]}}\\) 

Derivating Eq. 5a with respect to \\(a^{[L-1]}\\) :


 \\(\frac{\partial z^{[L]}}{\partial a^{[L-1]}} = \bar{w}^{[L]}\\)





### Eq. 5: Cost Derivatives for l=L-1

\\(\frac{\partial C}{\partial w^{[L]}}=\frac{\partial C}{\partial a^{[L]}}* \frac{\partial a^{[L]}}{\partial z^{[L]}}* \frac{\partial z^{[L]}}{\partial w^{[L]}}\\)








Let's examine layer L equations - look at Figure 2 (extracted from Figure 1).





To do that, we will use the derivative chain rule. 




denoted by l=L:

We need to find 

The cost function is Since  \\(\hat{y}=a^{[L]}\\) - see that in the last section of F_gure 1, _ \\(C(y,\hat{y})\\) is a function of \\(\hat{y})\\) i




















Eq. 1 presents a cost function general expression. It is denoted as a function of the network's coefficients w,b, which are the variable parameters, and g-the activation function, x - input data, and y-the expected output, which are given constants.


### Eq. 1: Cost Function

J(w,b,g,x,y) 






















