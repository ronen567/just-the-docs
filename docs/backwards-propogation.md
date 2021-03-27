---
layout: default
nav_order: 6
title: Introduction to Backwards Propogation
---
# Forward Propogation

## Introduction

This is the third in series of 3 deep learning intro posts:
1. Introduction to Deep Learning which introduces the Deep Learning technology background, and presents network's building blocks and terms.
2. Forward Propogation, which presents the mathematical equations of the prediction path.
3. Backward Propogation which presents the mathematical equations for network's coefficents calculation, done during the training phase.

In this post we will develop the equations for calculating the neural network's optimized weights and biases using backwards propogation.

## Cost Function and Gradient Descent
We have developped the neural networks Feead Forward equations, which calculates the predicted value \\(\hat{y}\\), based on input data \\(\bar{x}\\), for a given network structure, with a given set of weights and bias.

\\(\bar{Z}^{[l]}=\begin{bmatrix}z_1^{[l](1)}& z_1^{[l](2)} & . & . & z_1^{[l](m)}\\\\\\  
 z_2^{[l](1)}& z_2^{[l](2)} & . & . & z_2^{[l](m)}\\\\\\  
 z_3^{[l](1)}& z_3^{[l](2)} & . & . & z_3^{[l](m)}\\\\\\ 
 .& . & . & . &. \\\\\\ 
 . & . &.  & . & . \\\\\\ 
 z_n^{[l](1)}&z_n^{[l](2)}  & . & . & z_n^{[l](m)}\end{bmatrix}\\)


\\(\bar{A}^{[l]}=\begin{bmatrix}a_1^{[l](1)}& a_1^{[l](2)} & . & . & a_1^{[l](m)}\\\\\\  
 a_2^{[l](1)}& a_2^{[l](2)} & . &  .& a_2^{[l](m)}\\\\\\
 a_3^{[l](1)}& a_3^{[l](2)} & . & . & a_3^{[l](m)}\\\\\\ 
 .& . &  .&.  &. \\\\\\
 . & . & . & . & .\\\\\\ 
 a_n^{[l](1)}&a_n^{[l](2)}  & . & . & a_n^{[l](m)}\end{bmatrix}\\)

