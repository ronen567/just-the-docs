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

In prediction mode, as described in the previous post, the network and its coefficients are static, while only the input data changes. Given that, the predicted value is caculated using the forwarding equations presented. The Training AKA fitting mode is different: A set of example data is sent constantly, while the network's coefficients are modified till their values converge to an optimal value.
In this post we will develop the equations for calculating the neural network's optimized weights and biases using backwards propogation. 

## Cost Function and Gradeint Descent











