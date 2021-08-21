---
layout: default
title: Optimization Algorithms
nav_order: 4
---

## Introduction


**Gradient Descent** is a recursive algorithm, which finds the minimum of a function. The minimum is located by striding in the oposite direction of the function's gradient, with a step size according to the gradient size, as expressed in Eq. 1.

## Eq. 1: Gradient Descent Equation

\\(X(t) = X(t-1)-\alpha \cdot \bigtriangledown f(X)) \\)
**Where**:
**t** is the iteration index
\\(X = {x_j} j =1:n\\) the n dimensional parameter searched for.
\\(\alpha\\) is constant known as the learning rate.

The recursive algorithm goes as follows:

0. Init \\(X={x_j}\\), j=1:n to an arbitrary value. (A bad values selection may delay convergance. Check if so by trial and error.)
1. Calculate gradient of f(X)
2. Calculate new X(t) using Eq. 1
3. Continue to step 1 if \\(\bigtriangledown f(X) > \epsilon \\)


**Gradient Descent in Deep Neural Networks*** 
In the context Deep Neural Networks (DNNs), Gradient Descent is the most popular optimization algorithm, used to find the optimized set of weight coefficients which minimizes the cost function J, (i.e. a function which expresses the error between the expected DNN output and the model's predicted output). Figure 2 depicts a schematic diagram of DNNs Training phase functionality. 

![Training-Phase](../assets/images/gd_optimizations/Training-Phase.png)


**Illustrative Examples**

Figure 1 illustrates gradient descent convergence for a single variable function \\(f(x) = (x-w_1)^2\\). In this single variable example, the gradient degenerates to:
\\(x_{t} = x_{t-1}-\alpha \cdot \frac{d}{dx}f(x) \\)

![gradient decent example](../assets/images/gd_optimizations/sgd_1d_intro.gif)

Figure 2 is a contour graph which illustrates gradient descent convergence for a 2 variable function of the type \\(f(x) = a \cdot (x-w_1)^2 + b \cdot (x-w_1)^2\\)

**Figure 2: Gradient Descent Asymetric Gradients

![gradient decent example](../assets/images/gd_optimizations/2d.gif)


Now look at Figure 3, which is similar to figure 1 except that the gradient is stipper in one direction. That leads to some oscilations,but the gradient descent converges eventually.

**Figure 3: Gradient Descent Asymetric GRadients

![gradient decent example](../assets/images/gd_optimizations/2d_contour_sgd_asymetric.gif)


Figre 4 however presents an even stipper gradient in one direction. Now Gradient Descent oscilates in one direction and never converge. 

**Figure 4: Gradient Descent Oscilations
![gradient decent example](../assets/images/gd_optimizations/2d_contour_sgd_oscilations.gif)


Some phenomenas/Performance issues Gradient Descent suffers from are:
- Overshooting: As depicted by Figures 3 and 4, high curvutures may lead to overshooting. Overshooting is a result of moving too fast along the gradient direction, while it changes signs. 

- Local Minimun trap: Getting trapped in a local minimum, not reaching the global minima.

- Oscillations: this phenomenon can occure not only when gradient changes significantly in high curvuturs as depicted by Figure 4, but also when no matter the direction it  navigating in a plateau, where gradient is negligible but still may have slight differences which lead to oscliations

Overshooting could be avoided by setting a smaller learning rate, but that would slow the convergence process down.
Some of these phenomanas, e.g. overshooting, could be avoided by etting a smaller learning rate. But would have increased convergence time.
This post reviews some of the most popular Gradient Descent variants which aim to answer Gradient Descent issues.

This algorithms reviewd in this post as listed below:

**Momentum**
**Nesterov momentum**
**Adagrad**
**RMSprop**
**Adam**
**Adadelta**
**Adam**
**Adamax**
**FTR**
**NAdam**


## Momentum

The Momentum method aims to stabilize and thus accelerate Gradiennt Descent. As depicted in the above overshoot oscilation illustrated plots, step size would preferably become smaller inside the curveture, where gradient change directions, and would become larger when gradient sign does not change.  
The Momenentum method was first posted in the paper "Some Methods of speeding up convergence of iteration methods", USSR Computational Mathematics and Mathematical Physics, 1964 by B.T. Poylak. 

The Polyak momentum step adds a momentum term to the gradient descent fortmula:
\\(w(t+1)=w(t)-\alpha \cdot \bigtriangledown f(w(t)) + \beta(w(t)-w(t-1))\\)

The momentum term is the previous gradient descent step size, so intuition can already interpret that momenutum increases or decreases current step, if it is has the same direction or oposite direction, respectively.
\\(\beta\\) is a hyperparameter, often set to \\(\beta = 0.9\\).
So clearly, \\(w(t)-w(t-1))\\) is the value changed step in the previous iteration. If the current gradient is in the same direction, it has the same s \\(\\) if the gradinet \\(\cdot \bigtriangledown f(w(t\\) is negative, 


Figure 5 depicts a plot of Figure 4 oscliative Gradient Descent, now with Momentum.


### Figure 5: Momentum



**Nesterov momentum**

Nesterov momentum algorithm is a variation of different from Polyak, 


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
