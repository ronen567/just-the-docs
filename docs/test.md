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

## Cost Function and Gradient Descent
We have developped the neural networks Feead Forward equations, which calculates the predicted value \\(\hat{y}\\), based on input data \\(\bar{x}\\), for a given network structure, with a given set of weights and bias.

