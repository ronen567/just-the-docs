---
layout: default
title: Optimization Algorithms
nav_order: 2
---

# Optimization Algorithms - Gradient Descent Variations Algorithms

## Introduction

This post reviews commonly used variations of Gradient Descent algorithm. 
What Gradient Descent is, and why is it needed for Deep Neural Network (DNNs)? 
In short, suppose you have a parametarized function, say:
\\
Gradient Descent is a recursive optimization algorithm which is used to optimize Deep Neural Networks, by fitting it with the set of weight coefficients which minimize the output of the given cost function. Cost Function, btw, expresses the difference between the network's predicted value and the real expected value). This optimization, is executed during the Training phase (ofcourse). 

Eq. 1.a presents the basic Gradient Descent formula. It is a recursive formula, where the value the of a coefficient \\(a coefficient \\)Gradient Descent is a recursive formula Eq. 1 presents the basic Gradient Descent formula, recursive coefficient updates:

### Eq. 1.a Gradient Descent Formula

\\(w_{t}=w_{t-1} -\alpha \cdot \triangledown L(w_{t-1})\\)

Where:
\\(w_{t}\\) is the set of resultant wheight coefficients, calculated at iteration t.
\\(w_{t-1}\\) is the set of wheight coefficients, calculated at iteration t-1.
\\)\triangledown L(w_{t-1})
\\(\alpha\\) is the learning rate, which determines how much should the gradient effect the newly calculated value.

The Gradient Descent converges when the gradient reaches 0, i.e. at a minima.  
Here below are some graphical illustrations: Suppose we have m



This post reviews some of the Gradient Descent algorithms as listed below:

**Stochastic Gradient Descent**
**Momentum**
**Adagrad**
**RMSprop**
**Adam**
**tensorflow**
**SGD algorithm**
**SGD with Momentum algorithm**
**Adagrad algorithm**
**Adadelta algorithm**
**Adam algorithm**
**Adamax algorithm**
**FTRL algorithm**
**NAdam algorithm**
**RMSprop algorithm**


## Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) is the Gradient Descent variant which runs a coeffcients update iteration, using Eq. 1, for every single training example. 
Why is it called Stochastic? Because the gradient is calculated per a single sample, which is expected to consist of random noise. The noisy gradient slows down convergence, as a result of deflections from the path on the gradient's slope towards the gradient's minima, the progress towards the optimum is more hesitant and slow.

This is as oposed to Batch algoritms, AKA Deterministic Methods, which gradients used for Eq. 1 are the average value of m samples. Accordingly, the gradients are less noisy, i.e. more deterministic.
Still, SGD proves to fastly converge. However that, current High Processing Computing (HPC) devices, e.g. GPUs, are tailored for vectorized computations, and are not efficients for scalar computations.







## Momentum

The Momentum method aims to stabilize and thus acceleration of learning, in the following scenarios:

- High curvutures - High curvutures may lead to overshooting the Gradient Descent updates as depicted in Figure 1. 

- Overshooting as a result of moving too fast along the gradient direction: Overshooting could be avoided by setting a smaller learning rate, but that would slow the convergence process down.
- Local Minimun trap: Getting trapped in a local minimum, which is a direct consequence of SGD algorithm being greedy.

-Oscillation, this is a phenomenon that occurs when the function's value doesn't change significantly no matter the direction it advances. You can think of it as navigating a plateau, you're at the same height no matter where you go

Figure 1 illustrates graphically the effect of momentum on SGD convergence. The predictor here has 2 coefficients:

\\(\hat{y}=w_1x+b\\)

(Number of coefficients was limitted to permit the graphical presentation, but results can be generalized to any number of prediction dimensions.)

Figure 1a presents SGD with oscilation - the gradient in the y axis is stip, so the gradient update is oscilating from side to side, never converging. A small er learning rate might have avoid the oscilations, but on the other hand, that would have slowed down conversion.

Instead...



### Figure 1: The effect of momentum on SGD convergence

### Figure 1a: Plain SGD

![Oscilating SGD](../assets/images/gd_optimizations/sgd-oscilations.gif)



### Figure 1b: SGD with Momentum

![Momentum SGD](../assets/images/gd_optimizations/sgd-momentum.gif)





graph is limitted 


## Adagrad
## RMSprop
## Adam
tensorflow
SGD algorithm
SGD with Momentum algorithm
Adagrad algorithm.
Adadelta algorithm.
Adam algorithm.
Adamax algorithm.
FTRL algorithm.
NAdam algorithm.
RMSprop algorithm.









used for finding that set of minimizing coefficents. tech for finding the set of minimizing parameters is Gradient Descent, and function parameters are fitted during training, by minimizing a cost function which determines the error between the expected and predicted values. To most commonly used algorithm for finding the  minimum, is Gradient Descent, and its various variations. This post reviews the various optimization algorithms.







fare traine, is by minimizing a cost functionGradieThe most common method to train a neural network is by using gradient descent (SGD). The way this works is you define a loss function 
that expresses how well your weights & biases allow the network to fit your training data. A higher loss means that the network is bad and makes a lot of errors while a low loss generally means that the network performs well. You can then train your network by adjusting the network parameters in a way that reduces the loss.



coursera:
Mini Batch


Training with batch of all m examples - 1 update (step of gradient descent) after each cycle
Mini Batch - 1000 examples...runs faster for big training set


Referances:
- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville 
