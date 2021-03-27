---
layout: default
nav_order: 6
title: Introduction to Backwards Propogation
---
# Backwards Propogation

## Introduction

This is the third in series of 3 deep learning intro posts:
1. Introduction to Deep Learning which introduces the Deep Learning technology background, and presents network's building blocks and terms.
2. Forward Propogation, which presents the mathematical equations of the prediction path.
3. Backward Propogation which presents the mathematical equations for network's coefficents calculation, done during the training phase.

In this post we will develop the equations for calculating the neural network's optimized weights and biases using backwards propogation.


$$
\begin{bmatrix}
 &  & \\ 
 &  & 
\end{bmatrix}
$$

## Cost Function and Gradient Descent
We have developped the neural networks Feead Forward equations, which calculates the predicted value \\(\hat{y}\\), based on input data \\(\bar{x}\\), for a given network structure, with a given set of weights and bias.

$$
\begin{bmatrix}
 z_2^{[l](1)} &  z_2 & \\ 
 &   & 
\end{bmatrix}
$$


Eq.7 matrix dimenssions are:

 - \\(\bar{Z}^{[l]}\\) : m x n(l)
 - \\(\bar{w}^{[l]}\\) : n(l-1) x n(l)
 - \\(\bar{A}^{[l-1]}\\) : m x n(l-1)
 - \\(\bar{b}^{[l]}\\) : 1 x n(l)


Where n(l) is the number of neurons in layer l, and 

Note that the dimensions of the first term of Eq. 7a is  m x n(l-1), so the 1 x n(l) vector \\(\bar{b}^{[l]}\\) is added to it using broadcasting addition, i.e. \\(\bar{b}^{[l]}\\) is duplicated m times.

### Next steps
 
The next post in this series is about Backwards propogation, which is activated durimg the traing phase, aka fitting, to calculate optimized values for the network's wheights and biases.


