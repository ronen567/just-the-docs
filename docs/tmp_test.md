---
layout: default
title: test tmp
nav_order: 11
---
# test tmp and Minibatch
## Introduction

Gradient Descent and its variations are the most common algorithms used for fitting the DNN model during the Training phase. 

The basic formula of Gradient Descent parameter update is presented in Eq. 1:

### Eq. 1 Gradient Descent

\\(w=w-\alpha \cdot \frac{\partial L(w)}{\partial w}\\)

Where \\(L(w)\\) is a Loss function. Loss function expresses the prediction's accuracy calculated for a every Training data example.

Eq. 1 shows the formula for the recursive update of the network weight coefficient w, by substructing the Loss value  \\(L(w)\\) multipied by learning rate \\(\alpha\\).
\\(L(w)\\) is calculated per each Training data example. This variant of Gradient Descent is called **Stochastic Gradient Descent**, abrivated to **SGD**. Naturally, The gradients of the Loss function may be noisy, as a result of the random noise normally added to the input data samples.

The noisy gradient slows down convergence, as a result of deflections from the path on the sgradient's slpe towards the gradient's minima.
Calculating an averaged value of the gradient, over a batch of samples, may solve this slow down in convergence. The rest of this post introduces the considerations involved in selecting the batch size.
